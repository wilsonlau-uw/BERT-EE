import itertools
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from config import Config
import util
from logger import Logger
import sklearn
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.simplefilter(action='ignore', category=sklearn.exceptions.UndefinedMetricWarning)
import logging
logging.getLogger('transformers').setLevel(level=logging.ERROR)
import os
from os import listdir
from os.path import isfile, join, splitext
from string import punctuation
from collections import Counter, defaultdict
import json

import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from seqeval.metrics import classification_report
from transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                          BertForTokenClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)
from torch import nn


class Dropout(torch.nn.Module):
    def __init__(self, hidden_dropout_prob):
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, outputs):
        pooled_output = self.dropout(outputs[1])
        return pooled_output

class RE_Dataset(Dataset):

    def __init__(self,  all_samples, tokenizer, unique_labels=None):
        self.__all_samples = all_samples
        self.__tokenizer = tokenizer
        self.__max_seq_len = Config.getint('model','max_seq_len')

        self.__input_ids = []
        self.__attention_mask = []
        self.__token_type_ids = []
        self.__labels = []
        self.__position_ids = []
        self.__masks = []
        self.bert_tokens = []
        self.tokens = []
        self.token_indices=[]
        self.labels =[]
        self.sample_indices = []
        tokens_count=[]

        self.file_names=[]

        if unique_labels is not None:
            self.unique_labels = unique_labels
        else:
            self.unique_labels = list(set([s['relation'] for s in all_samples]))
            self.unique_labels.sort()

        self.label2id_map = {t: i for i, t in enumerate(self.unique_labels)}
        self.id2label_map = {i: t for i, t in enumerate(self.unique_labels)}

        for i, sample in enumerate(tqdm(all_samples)):

            h_pos = sample['h']['pos']
            t_pos = sample['t']['pos']

            bert_tokens = ['[CLS]']
            for j, token in enumerate(sample['token']):
                word_pieces = self.__tokenizer.tokenize(token)
                if j==h_pos[0]:
                    bert_tokens.append('[unused0]')
                if j==t_pos[0]:
                    bert_tokens.append('[unused1]')
                bert_tokens.extend(word_pieces)
                if j==h_pos[1]-1:
                    bert_tokens.append('[unused2]')
                if j==t_pos[1]-1:
                    bert_tokens.append('[unused3]')

                if j == self.__max_seq_len - 2:
                    break

            tokens_count.append(len(bert_tokens))

            if len(bert_tokens) >= self.__max_seq_len - 1:
                bert_tokens = bert_tokens[0:(self.__max_seq_len - 1)]

            bert_tokens.append("[SEP]")
            input_ids = self.__tokenizer.convert_tokens_to_ids(bert_tokens)
            attention_mask = [1] * len(input_ids)
            position_ids = [1] * len(input_ids)
            masks = [1] * len(input_ids)

            # padding
            padding_len = self.__max_seq_len - len(input_ids)
            input_ids.extend([0] * padding_len)
            attention_mask.extend([0] * padding_len)
            position_ids.extend([1] * padding_len)
            masks.extend([0] * padding_len)
            token_type_ids = [0] * self.__max_seq_len


            assert len(input_ids) == self.__max_seq_len
            assert len(attention_mask) == self.__max_seq_len
            assert len(token_type_ids) == self.__max_seq_len
            assert len(position_ids) == self.__max_seq_len

            self.__input_ids.append(input_ids)
            self.__attention_mask.append(attention_mask)
            self.__token_type_ids.append(token_type_ids)
            self.__position_ids.append(position_ids)
            self.bert_tokens.append(bert_tokens)
            self.__masks.append(masks)
            self.tokens.append(sample['token'])
            self.labels.append(sample['relation'])
            self.token_indices.append([0] * self.__max_seq_len) # not used
            self.sample_indices.append(sample['sample_idx'])
            self.file_names.append(sample['filename'])

            if sample['relation'] in self.label2id_map.keys():
                self.__labels.append(self.label2id_map[sample['relation']])
            else:
                self.__labels.append(self.label2id_map['no_relation'])

        Logger.info('# of tokens min: {} mean: {:.2f} median: {:.2f} max: {}'.format(np.min(tokens_count) if tokens_count.__len__()>0 else 0,
                                                                           np.mean(tokens_count) if tokens_count.__len__()>0 else 0,
                                                                           np.median(tokens_count) if tokens_count.__len__()>0 else 0,
                                                                           np.max(tokens_count) if tokens_count.__len__()>0 else 0))

        cnt=Counter(self.labels)
        for key, value in cnt.items():
            Logger.info('# of {} : {}'.format(key if key is not None else 'unlabelled relations' , value))




    def __len__(self):
        return len(self.__input_ids)

    def __getitem__(self, index):

        data = (torch.tensor(self.__input_ids[index]),
                torch.tensor(self.__token_type_ids[index]),
                torch.tensor(self.__attention_mask[index]),
                torch.tensor(self.__position_ids[index]),
                torch.tensor(self.__labels[index]) ,
                torch.tensor(self.__masks[index]), index)
        return data


class RE_encoder():

    def __init__(self,  bert_config):

        self.__dropout = Config.getfloat("re","dropout" ) 
        self.__fine_tuned_path = Config.getstr("general","fine_tuned_path")
        self.__distance = Config.getint("re","distance", usefallback=True, fallback= None)
        self.__num_workers = Config.getint("re","num_workers", usefallback=True, fallback= None)
        self.__no_relation_ratio = Config.getfloat("re","no_relation_ratio", usefallback=True, fallback= None)
        self.__bert_config = bert_config
        self.event_entities = {}
        self.relations_map=defaultdict(list)

        from brat_interface import BRAT_interface
        self.brat = BRAT_interface()

    def build(self, tokenizer,use_fine_tuned,  predict=False):

        self.__best_results = {'acc_all': 0, 'precsions_all': 0, 'recalls_all': 0, 'f1_all': 0,
                               'acc_labels': 0, 'precsions_labels': 0, 'recalls_labels': 0, 'f1_labels': 0, }

        self.__tokenizer = tokenizer
        load_test=False
        load_train=False
        load_validate= False

        if use_fine_tuned:
            model_data = json.load(open(os.path.join(self.__fine_tuned_path, 'model.json'), 'r'))
            if 're' not in model_data:
                Logger.error('The model ('+self.__fine_tuned_path+') may not be trained for RE.')

            self.unique_labels = model_data['re']['labels']
            self.training_size = model_data['re']['training_size']
            self.event_entities = model_data['re']['event_entities']
            self.relations_map = json.loads(model_data['re']['relations_map'])
            load_test = predict
            load_validate = not predict

        elif predict:
            load_test = load_train = load_validate = True
        else:
            load_train = load_validate = True

        if load_train:
            Logger.info('reading RE training data in folder {}'.format(Config.getstr('re','training_folder')))
            train_data = self.brat.read_brat_events(Config.getstr('re','training_folder'),
                                                            entities_labels=Config.getlist('ner','labels',usefallback=True),
                                                            relation_labels=Config.getlist('re','labels',usefallback=True),
                                                            distance=self.__distance,unlabelled=False,
                                                            no_relation_ratio=self.__no_relation_ratio,
                                                            max_sent_windows=Config.getint('re', 'max_sent_windows',
                                                                              usefallback=True),
                                                            num_workers=self.__num_workers)

            for t in train_data:
                ents = t['h']['label']+'|'+t['t']['label']
                if ents not in self.event_entities and t['relation']!='no_relation':
                    self.event_entities[ents]=set()

                if t['relation']!='no_relation':
                    self.event_entities[ents].add(t['relation'])

            self.relations_map = self.event_entities.copy()
            for k in self.relations_map:
                self.relations_map[k] = list(self.relations_map[k])



            with open(os.path.join(Config.get_resultfolder('re'), 'training.txt'), 'w') as wf:
                for t in train_data:
                    wf.write(json.dumps(t)+'\n')

            if len(train_data)==0:
                Logger.error ('No RE training data in '+Config.getstr('re','training_folder' ))

            Logger.info('tokenize RE tranining data')

            self.training_dataset = RE_Dataset(  train_data, self.__tokenizer)
            if self.training_dataset.sample_indices.__len__()==0:
                Logger.error('RE training data contains no events')

            self.training_size = len(self.training_dataset)
            self.unique_labels = self.training_dataset.unique_labels

        if load_validate:
            Logger.info('reading RE validation data')
            validate_relations_map=defaultdict(list)
            self.validate_data = self.brat.read_brat_events(Config.getstr('re','validation_folder'),
                                                       entities_labels=Config.getlist('ner', 'labels', usefallback=True),
                                                       relation_labels=Config.getlist('re', 'labels', usefallback=True),
                                                           distance=self.__distance,unlabelled=False,
                                                       # relations_map=validate_relations_map,
                                                            no_relation_ratio=self.__no_relation_ratio,
                                                           num_workers=self.__num_workers)

            for t in self.validate_data:
                ents = t['h']['label']+'|'+t['t']['label']
                if t['relation']!='no_relation':
                    if ents not in self.relations_map :
                        self.relations_map[ents]=[t['relation']]
                    elif t['relation'] not in self.relations_map[ents]:
                        self.relations_map[ents].append(t['relation'])

            with open(os.path.join(Config.get_resultfolder('re'), 'validation.txt'), 'w') as wf:
                for t in self.validate_data:
                    wf.write(json.dumps(t)+'\n')

            Logger.info('tokenize RE validation data')
            self.validation_dataset = RE_Dataset(  self.validate_data, self.__tokenizer,
                                                  unique_labels=self.unique_labels)

            if self.validation_dataset.sample_indices.__len__()==0:
                Logger.error('RE validation data contains no events')

        self.__label2id_map = {t: i for i, t in enumerate(self.unique_labels)}
        self.__id2label_map = {i: t for i, t in enumerate(self.unique_labels)}

        self.dropout = Dropout(self.__dropout) #Dropout(self.__bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.__bert_config.hidden_size, len(self.unique_labels))

    def get_test_data_total(self,folder):
        return self.brat.get_brat_unlabelled_files(folder).__len__()

    def get_test_data(self, folder):

        for self.__unlabelled_data, self.__unlabelled_file in self.brat.read_brat_unlabelled_events(folder,
                                                                       event_entities=self.event_entities,
                                                                       distance=self.__distance,
                                                                       max_sent_windows=Config.getint('re', 'max_sent_windows',usefallback=True),
                                                                       num_workers=self.__num_workers):

            with open(os.path.join(Config.get_resultfolder('re'), 'test.txt'), 'a') as wf:
                for t in self.__unlabelled_data:
                    wf.write(json.dumps(t)+'\n')

            self.unlabelled_dataset = RE_Dataset(self.__unlabelled_data, self.__tokenizer,
                                                  unique_labels=self.unique_labels)
            yield self.unlabelled_dataset

    def criterion(self, labels, attention_mask, logits):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, len(self.unique_labels)), labels.view(-1))

        return loss

    def start(self):
        self.__y_true = []
        self.__y_pred = []
        self.__pred_logits=[]
        self.__pred_sample_idx=[]

    def update(self, sample_idx, filenames, bert_labels, preds, bert_tokens, tokens, tokens_indices, logits, labels=None):

        self.__y_true.extend(bert_labels)
        self.__y_pred.extend(preds)
        self.__pred_logits.extend(logits)
        self.__pred_sample_idx.extend(sample_idx)

    def end(self):
        return  self.__y_true, self.__y_pred

    def show_best_result(self):
        self.show_evaluation_result( self.__best_results['acc_all'], self.__best_results['precsions_all'],
                               self.__best_results['recalls_all'], self.__best_results['f1_all'],
                                     self.__best_results['acc_labels'], self.__best_results['precsions_labels'],
                                     self.__best_results['recalls_labels'], self.__best_results['f1_labels']
                                     )

        for i,t in enumerate(self.__y_true):
            self.validate_data[self.__pred_sample_idx[i]]['y_pred']=self.__id2label_map[self.__y_pred[i]]
            self.validate_data[self.__pred_sample_idx[i]]['y_true'] = self.__id2label_map[self.__y_true[i]]

        util.saveData(self.validate_data, os.path.join(Config.get_resultfolder('re'), 'evaluation.pickle') )

    def show_evaluation_result(self,acc_all, precsions_all, recalls_all, f1_all,
                                    acc_labels, precsions_labels, recalls_labels, f1_labels, labels=None):

        if labels is None:
            y_true=[self.__id2label_map[l] for l in self.__y_true]
            labels = set(y_true)
            labels = sorted(labels)

        Logger.info('(labels only)    accuracy: {:.2f}%; precision: {:.2f}%; recall: {:.2f}%; FB1: {:.2f}% '.format(
            acc_labels * 100,
            np.mean(precsions_labels) * 100,
            np.mean(recalls_labels) * 100,
            np.mean(f1_labels) * 100))

        Logger.info('(w/ no_relation) accuracy : {:.2f}%; precision: {:.2f}%; recall: {:.2f}%; FB1: {:.2f}% '.format(acc_all * 100,
                                                                                             np.mean(precsions_all) * 100,
                                                                                             np.mean(recalls_all) * 100,
                                                                                             np.mean(f1_all) * 100))


        max_label_len = max([len(l) for l in labels])
        fmt = '\t{{:<{}s}}\tprecision:{{:>{},.2f}}%  recall: {{:>{},.2f}}%  FB1: {{:>{},.2f}}%'.format(max_label_len+2,7, 7,7)  # widths only

        for i, l in enumerate(labels):
            Logger.info(  fmt.format( labels[i], precsions_all[i] * 100,  recalls_all[i] * 100,  f1_all[i] * 100))

    def generate_predictions(self):
        if len(self.__unlabelled_file) > 0:
            data =[]
            preds=[]
            for i, d in enumerate(self.__unlabelled_data):
                key =d['h']['label']+'|'+d['t']['label']
                y_pred=self.__id2label_map[self.__y_pred[i]]
                data.append(d)
                if key in self.relations_map and  (y_pred in(self.relations_map[key]) or y_pred == 'no_relation'):
                    preds.append(y_pred)
                else:
                    preds.append('no_relation')

            t2h_map = defaultdict(list)
            event_heads=[h.split('|')[0] for h in self.event_entities]
            linked_entities = set()
            for i, p in enumerate(preds):
                en_key = '{}-{}-{}'.format(data[i]['t']['name'], data[i]['t']['start_idx'], data[i]['t']['end_idx'])
                if p != 'no_relation' :
                    linked_entities.add(en_key)

            for i, p in enumerate(preds):
                en_key = '{}-{}-{}'.format(data[i]['t']['name'], data[i]['t']['start_idx'], data[i]['t']['end_idx'])
                if p == 'no_relation' and '{}|{}'.format(data[i]['h']['label'], data[i]['t'][
                    'label']) in self.event_entities \
                        and (data[i]['t']['label'] not in event_heads) \
                        and en_key not in linked_entities:
                    t2h_map[en_key].append(data[i])

            for t in t2h_map:
                candidate_hs = sorted(t2h_map[t], key=lambda x: abs(x['h']['start_idx'] - x['t']['start_idx']))
                preds[candidate_hs[0]['sample_idx']] = \
                self.relations_map['{}|{}'.format(candidate_hs[0]['h']['label'], candidate_hs[0]['t']['label'])][0]

            self.brat.save_brat_re(self.__unlabelled_file, Config.get_resultfolder('re'), data,preds)

    def evaluate(self):

        Logger.info("***** Eval RE results *****")

        self.__y_true_all=[self.__id2label_map[l] for l in self.__y_true]
        self.__y_pred_all= [self.__id2label_map[l] for l in self.__y_pred]

        y_true_labels =[]
        y_pred_labels =[]
        for i,y in enumerate(self.__y_true_all):
            if y != 'no_relation':
                y_true_labels.append(y)
                y_pred_labels.append(self.__y_pred_all[i])

        corre = np.sum([g == self.__y_pred_all[i] for i, g in enumerate(self.__y_true_all)])
        self.__acc_all = corre / len(self.__y_true_all)
        corre = np.sum([g == y_pred_labels[i] for i, g in enumerate(y_true_labels)])
        self.__acc_labels = corre / len(y_true_labels)

        labels = set(self.__y_true_all)
        labels = sorted(labels)
        self.__precsions_all, self.__recalls_all, self.__f1_all, _ = precision_recall_fscore_support(self.__y_true_all, self.__y_pred_all, labels=labels)
        self.__precsions_labels, self.__recalls_labels, self.__f1_labels, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels, labels=[l for l in labels if l != 'no_relation'])
        self.show_evaluation_result(self.__acc_all, self.__precsions_all, self.__recalls_all, self.__f1_all,
                                    self.__acc_labels, self.__precsions_labels, self.__recalls_labels, self.__f1_labels,labels)

        return self.__acc_all, np.mean(self.__precsions_all), np.mean(self.__recalls_all), np.mean(self.__f1_all)

    def save_best_results(self):

        self.__best_results = {'acc_all': self.__acc_all, 'precsions_all': self.__precsions_all, 'recalls_all': self.__recalls_all,
                               'f1_all': self.__f1_all,
                               'acc_labels': self.__acc_labels, 'precsions_labels': self.__precsions_labels,
                               'recalls_labels': self.__recalls_labels, 'f1_labels': self.__f1_labels}

        util.saveData([self.__y_true_all, self.__y_pred_all],
                  os.path.join(Config.get_resultfolder('re'), 're_validations.pkl'))

    def model_meta_data(self):

        return  {'training_size': self.training_size,
                'labels': self.training_dataset.unique_labels,
                 'event_entities' : list(self.event_entities.keys()),
                 'relations_map': json.dumps(self.relations_map)}


