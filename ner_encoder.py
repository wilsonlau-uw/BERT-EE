import itertools
from config import Config
import numpy as np
import os
from tqdm import tqdm
import util
from logger import Logger
import logging
logging.getLogger('transformers').setLevel(level=logging.ERROR)
import json
import torch
from torch.utils.data import Dataset,RandomSampler,SequentialSampler,DataLoader
from seqeval.metrics import classification_report
from torch import nn


class Dropout(torch.nn.Module):
    def __init__(self, hidden_dropout_prob):
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, outputs):
        sequence_output = self.dropout(outputs[0])
        return sequence_output


class NER_Dataset(Dataset):

    def __init__(self,  all_samples, tokenizer, unique_labels=None):
        self.__all_samples = all_samples
        self.__tokenizer = tokenizer
        self.__max_seq_len = Config.getint("model","max_seq_len")
        self.__input_ids = []
        self.__attention_mask=[]
        self.__token_type_ids=[]
        self.__labels=[]
        self.__position_ids=[]
        self.__masks=[]
        self.bert_tokens = []
        self.tokens = []
        self.token_indices=[]
        self.labels =[]
        self.sample_indices=[]
        self.file_names=[]

        tokens_count=[]

        if unique_labels is not None:
            self.unique_labels = unique_labels
        else:
            self.unique_labels = ['[PAD]']+list(set(list(itertools.chain.from_iterable([s['labels'] for s in all_samples])))) +['#','[CLS]','[SEP]','[UNK]']

        self.label2id_map = {t : i for i, t in enumerate(self.unique_labels)}
        self.id2label_map = {i: t  for i, t in enumerate(self.unique_labels)}

        for i, samples in enumerate(tqdm(all_samples)):
            labels = [s for s in samples['labels']]
            bert_tokens = ['[CLS]']
            bert_labels = [self.label2id_map["[CLS]"]]
            masks = [1]
            position_ids = [1]
            token_indices=[]

            for z , s in enumerate(labels):
                if(s!='O'):
                    Logger.debug(samples['filename'], samples['tokens'][z],s,z)

            for j, token in enumerate(samples['tokens'] ):
                word_pieces = self.__tokenizer.tokenize(token)
                bert_tokens.extend(word_pieces)
                token_indices.append(samples['indices'][j])
                for k, wp in enumerate(word_pieces):
                    if k == 0:
                        if(len(labels)>0):
                            if labels[j] in self.label2id_map:
                                bert_labels.append(self.label2id_map[labels[j]])
                            else:
                                bert_labels.append(self.label2id_map['[UNK]'])
                        position_ids.append(1)
                    else:
                        if (len(labels) > 0):
                            bert_labels.append(self.label2id_map['#'])
                        position_ids.append(0)

            tokens_count.append(len(bert_tokens))

            if len(bert_tokens) >= self.__max_seq_len - 1:
                bert_tokens = bert_tokens[0:(self.__max_seq_len - 1)]
                bert_labels = bert_labels[0:(self.__max_seq_len - 1)]
                position_ids = position_ids[0:(self.__max_seq_len - 1)]

            bert_tokens.append("[SEP]")
            position_ids.append(1)
            masks.append(1)
            bert_labels.append(self.label2id_map["[SEP]"])
            input_ids = self.__tokenizer.convert_tokens_to_ids(bert_tokens)
            attention_mask = [1] * len(input_ids)
            masks = [1] * len(input_ids)
            token_type_ids = [0] * self.__max_seq_len

            # padding
            padding_len = self.__max_seq_len - len(input_ids)
            input_ids.extend([0] * padding_len)
            attention_mask.extend([0] * padding_len)
            bert_labels.extend([0] * (self.__max_seq_len - len(bert_labels)))
            position_ids.extend([1] * padding_len)
            masks.extend([0] * padding_len)
            token_indices.extend([0] * (self.__max_seq_len - len(token_indices)))

            assert len(input_ids) == self.__max_seq_len
            assert len(attention_mask) == self.__max_seq_len
            assert len(token_type_ids) == self.__max_seq_len
            assert len(position_ids) == self.__max_seq_len
            assert len(masks) == self.__max_seq_len
            assert len(bert_labels) == self.__max_seq_len

            self.__input_ids.append(input_ids)
            self.__attention_mask.append(attention_mask)
            self.__token_type_ids.append(token_type_ids)
            self.__labels.append(bert_labels)
            self.__position_ids.append(position_ids)
            self.__masks.append(masks)
            self.bert_tokens.append(bert_tokens)
            self.tokens.append(samples['tokens'])
            self.labels.append(samples['labels'])
            self.token_indices.append(token_indices)
            self.sample_indices.append(samples['sample_idx'])
            self.file_names.append(samples['filename'])

        Logger.info('# tokens min: {} mean: {:.2f} median: {:.2f} max: {}'.format(np.min(tokens_count),
                                                                           np.mean(tokens_count),
                                                                           np.median(tokens_count),
                                                                           np.max(tokens_count)))

    def __len__(self):
        return len(self.__input_ids)

    def __getitem__(self,index):

        data = (torch.tensor(self.__input_ids[index]), torch.tensor(self.__token_type_ids[index]),
                 torch.tensor(self.__attention_mask[index]),    torch.tensor(self.__position_ids[index]),
                 torch.tensor(self.__labels[index]),
                torch.tensor(self.__masks[index]) , index)
        return data

class NER_encoder():

    def __init__(self,   bert_config):

        self.__dropout =  Config.getfloat("ner","dropout")
        self.__use_fine_tuned = Config.getboolean("general", "use_fine_tuned")
        self.__fine_tuned_path= Config.getstr("general","fine_tuned_path")
        self.__bert_config=bert_config

        self.__y_true_sp = []
        self.__y_pred_sp = []
        self.__all_bert_tokens = []
        self.__all_tokens = []
        self.__all_tokens_indices=[]
        self.__unlabelled_file =None
        self.__softlabels = None

        from brat_interface import BRAT_interface
        self.brat = BRAT_interface()

    def build(self,tokenizer,use_fine_tuned,  predict=False):

        self.__best_predictions={}
        self.__best_results = {'acc': 0, 'precsions': 0, 'recalls': 0, 'f1': 0}
        self.__tokenizer = tokenizer
        load_test=False
        load_train=False
        load_validate= False

        if use_fine_tuned:
            model_data = json.load(open(os.path.join(self.__fine_tuned_path, 'model.json'), 'r'))
            if 'ner' not in model_data:
                Logger.error('The model may not be trained for NER.')

            self.unique_labels = model_data['ner']['labels']
            self.training_size = model_data['ner']['training_size']

            load_test=predict
            load_validate=not predict

        elif predict:
            load_test=load_train=load_validate=True
        else:
            load_train = load_validate = True

        if load_train:

            Logger.info('reading NER training data in folder {}'.format(Config.getstr('ner','training_folder')))
            train_data = self.brat.read_brat(Config.getstr('ner','training_folder'),
                                        Config.getlist('ner','labels',usefallback=True),
                                        Config.getstr('ner','scheme'),allow_overlap=False,use_sent_segmentation=Config.getboolean('general','use_sent_segmentation'))

            Logger.info('tokenize NER training data')
            self.training_dataset = NER_Dataset(  train_data, self.__tokenizer)
            self.training_size = len(self.training_dataset)
            self.unique_labels = self.training_dataset.unique_labels

        if load_validate:
            Logger.info('reading NER validation data in folder {}'.format(Config.getstr('ner','validation_folder')))
            labels_count={}
            validate_data = self.brat.read_brat(Config.getstr('ner','validation_folder'),Config.getlist('ner','labels',usefallback=True),
                                           Config.getstr('ner','scheme'),allow_overlap=False,use_sent_segmentation=Config.getboolean('general','use_sent_segmentation'))

            Logger.info('tokenize NER validation data')
            self.validation_dataset = NER_Dataset( validate_data, self.__tokenizer,
                                                    unique_labels=self.unique_labels)

        if load_test:
            self.__prediction_folder = Config.getstr('ner','prediction_folder')

        self.__label2id_map = {t: i for i, t in enumerate(self.unique_labels)}
        self.__id2label_map = {i: t for i, t in enumerate(self.unique_labels)}

        self.dropout = Dropout(self.__dropout)
        self.classifier = nn.Linear(self.__bert_config.hidden_size, len( self.unique_labels))

    def get_test_data_total(self):
        return self.brat.get_brat_unlabelled_files(self.__prediction_folder).__len__()

    def get_test_data(self):

        for (self.__unlabelled_data, self.__unlabelled_text) in self.brat.read_brat_unlabelled_entities(self.__prediction_folder,Config.getboolean('general','use_sent_segmentation')):
            self.__unlabelled_file = self.__unlabelled_data[0]['filename']
            # Logger.info('predicting entities in ', self.__unlabelled_file)
            self.unlabelled_dataset = NER_Dataset(self.__unlabelled_data, self.__tokenizer,
                                                  unique_labels=self.unique_labels)
            yield self.unlabelled_dataset

    def criterion(self,labels,attention_mask, logits):

        loss_fct = nn.CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, len(self.unique_labels))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, len(self.unique_labels)), labels.view(-1))

        return loss

    def start(self):
        self.__y_true_sp = []
        self.__y_pred_sp = []
        self.__all_bert_tokens = []
        self.__all_tokens = []
        self.__all_tokens_indices=[]
        self.__all_labels=[]

        self.__preds=[]
        self.__pred_logits=[]
        self.__pred_sample_idx=[]
        self.__all_filenames = []

    def update(self,sample_idx,filenames,bert_labels, preds,bert_tokens,tokens,tokens_indices,logits,labels=None):

        if bert_labels is not None:
            self.__map_predict_labels(self.__y_pred_sp, preds,bert_tokens, True)
        else:
            self.__map_labels(self.__y_true_sp, self.__y_pred_sp, bert_labels, preds, True)

        self.__all_bert_tokens.extend(bert_tokens)
        self.__all_tokens.extend(tokens)
        self.__all_tokens_indices.extend(tokens_indices)
        self.__preds.extend(preds)
        self.__pred_sample_idx.extend(sample_idx)
        self.__pred_logits.extend(logits)
        self.__all_filenames.extend(filenames)

        if labels is not None:
            self.__all_labels.extend(labels)


    def end(self):

        self.__output_tokens, self.__output_indices, self.__predicted_labels_flattened, self.__true_labels_flattened,\
        self.__predicted_labels , self.__true_labels ,  \
            = self.__compile_results(self.__y_pred_sp, self.__all_bert_tokens, self.__all_tokens,self.__all_tokens_indices, self.__all_labels)


        return  self.__true_labels, self.__predicted_labels

    def generate_predictions(self):

        self.brat.save_brat_ner( self.__unlabelled_file,
                           Config.get_resultfolder('ner'),
                           self.__predicted_labels,
                           self.__output_tokens,
                           self.__output_indices,
                           self.__unlabelled_text
                            , Config.getstr('ner', 'scheme'))

        util.saveData([self.__unlabelled_file,
                           Config.get_resultfolder('ner'),
                           self.__predicted_labels,
                           self.__output_tokens,
                           self.__output_indices,
                           self.__unlabelled_text], 'generate_predictions.pickle')

    def show_best_result(self):
        results_str, results = classification_report(self.__best_predictions['true_labels'],
                                                      self.__best_predictions['predicted_labels'],
                                                      scheme=Config.getstr('ner', 'scheme'),
                                                      mode=None if Config.getstr('ner',
                                                                                 'eval_mode') == 'None' else Config.getstr(
                                                          'ner', 'eval_mode'))
        Logger.info('\n'+results_str)

    def evaluate(self):

        Logger.info("***** Eval NER results *****")
        results_str,results=classification_report(self.__true_labels, self.__predicted_labels,output_dict=True,
                                       scheme=Config.getstr('ner','scheme'),
                                       mode=None if Config.getstr('ner','eval_mode')=='None' else Config.getstr('ner','eval_mode'))
        Logger.info('\n'+results_str)
        self.__prec = results['micro avg']['precision']
        self.__rec = results['micro avg']['recall']
        self.__f1 = results['micro avg']['f1-score']

        y_true =  self.__true_labels
        y_pred =  self.__predicted_labels
        corre = np.sum([g == y_pred[i] for i, g in enumerate(y_true)])
        self.__acc = corre / len(y_true)

        return self.__acc, self.__prec, self.__rec, self.__f1

    def model_meta_data(self):

        return  {'training_size': self.training_size,
                'labels': self.training_dataset.unique_labels}

    def save_best_results(self):

        self.__best_results = {'acc': self.__acc, 'precsions': self.__prec, 'recalls': self.__rec, 'f1': self.__f1}
        self.__best_predictions = {'true_labels': self.__true_labels, 'predicted_labels': self.__predicted_labels}

        util.saveData([self.__true_labels, self.__predicted_labels, self.__all_bert_tokens, self.__all_tokens,self.__output_tokens,self.__all_filenames], os.path.join(Config.get_resultfolder('ner'), 'ner_validations.pkl'))


    def __map_labels(self,y_true, y_pred, all_labels, preds, include_sp_char = False):

        for i, labels in enumerate(all_labels):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(labels):
                if j == 0:
                    continue
                elif all_labels[i][j] == self.__label2id_map['[SEP]']:
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                elif include_sp_char == True or self.__id2label_map[all_labels[i][j]] != '#':
                    temp_1.append(self.__id2label_map[all_labels[i][j]])
                    temp_2.append(self.__id2label_map[preds[i][j]])

    def __map_predict_labels(self, y_pred,   preds, all_bert_tokens, include_sp_char = False):

        for i, pred in enumerate(preds):
            temp_2 = []
            for j, m in enumerate(pred):
                if j == 0:
                    continue

                elif j == len(all_bert_tokens[i]) - 1 or j == len(pred):
                    y_pred.append(temp_2)
                    break
                elif include_sp_char == True or self.__id2label_map[pred[j]] != '#':
                    temp_2.append(self.__id2label_map[pred[j]])


    def __remove_special_tokens(self, tokens):
        temp = [t for t in tokens if t not in ['[PAD]', '#', '[CLS]']]
        if '[SEP]' in temp:
            idx = temp.index('[SEP]')
            return temp[:idx]
        else:
            return temp

    def __compile_results(self,all_preds, all_bert_tokens, all_tokens, all_tokens_indices, batch_labels):

        aligned_tokens = []
        aligned_indices = []
        aligned_preds = []
        bert_special_tokens = ["[PAD]", "[CLS]", "[SEP]", '#']

        for i, tokens in enumerate(all_tokens):
            bert_tokens = [t for t in all_bert_tokens[i] if t != "[PAD]" and t != "[CLS]" and t != "[SEP]"]
            bert_tokens = [t[2:] if t[0:2] == '##' else t for t in bert_tokens]
            indices = all_tokens_indices[i]
            preds = all_preds[i]

            if tokens.__len__() == len(bert_tokens):
                aligned_tokens.append(tokens)
                aligned_indices.append(list(indices[0:len(tokens)]))
                aligned_preds.append([p if p not in bert_special_tokens else 'O' for p in preds])

            elif len(tokens) == 1:
                aligned_tokens.append(tokens)
                aligned_indices.append([indices[0]])
                if len(preds)>0:
                    aligned_preds.append([preds[0] if preds[0] not in bert_special_tokens else 'O'])
                else:
                    aligned_preds.append(['O'])


            else:
                current_indices = []
                current_preds = []
                current_tokens = []
                token_s = 0
                token_e = 0
                for j, t in enumerate(tokens):
                    # if bert's seq max len is shorter than actual sequence
                    if token_s == bert_tokens.__len__():
                        current_indices.append(indices[j])
                        current_preds.append('O')
                        current_tokens.append(t)

                    else:

                        current_indices.append(indices[j])
                        current_preds.append(preds[token_s] if preds[token_s] not in bert_special_tokens else 'O')
                        current_tokens.append(t)
                        while ''.join(bert_tokens[token_s:token_e]).lower() != t.lower():
                            token_e += 1
                            if token_e == len(bert_tokens): break

                        token_s = token_e

                aligned_indices.append(current_indices)
                aligned_preds.append(current_preds)
                aligned_tokens.append(current_tokens)

        flattened_aligned_tokens = list(itertools.chain.from_iterable(aligned_tokens))
        flattened_aligned_indices = list(itertools.chain.from_iterable(aligned_indices))
        flattened_aligned_preds = list(itertools.chain.from_iterable(aligned_preds))
        flattened_batch_labels = list(itertools.chain.from_iterable(batch_labels))

        assert len(flattened_aligned_tokens) == len(flattened_aligned_preds)
        assert len(flattened_aligned_tokens) == len(flattened_aligned_indices)
        if batch_labels.__len__() > 0:
            assert len(flattened_aligned_tokens) == len(flattened_batch_labels)

        return flattened_aligned_tokens, flattened_aligned_indices, flattened_aligned_preds, flattened_batch_labels, aligned_preds, batch_labels



