
from config import Config
from logger import Logger
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from ner_encoder import NER_encoder
from re_encoder import RE_encoder
from multitask_dataset import MultiTaskDataset
import torch.nn as nn
from transformers import WEIGHTS_NAME, AdamW,  BertConfig,  BertTokenizer,  get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert  import BertPreTrainedModel, BertModel,BertForSequenceClassification
from torch.utils.data import Dataset,RandomSampler,SequentialSampler,DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR


class BERT_EE_model(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.classifiers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.init_weights()

    def add_classifier(self, classifier):
        self.classifiers.append(classifier)

    def add_dropout(self, dropout):
        self.dropouts.append(dropout)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            task_id=None,
            num_labels=None

    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        output = self.dropouts[task_id](outputs)
        logits = self.classifiers[task_id](output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs


class BERT_EE(nn.Module):

    def __init__(self):
        super(BERT_EE, self).__init__()
        self.use_fine_tuned = Config.getboolean("general", "use_fine_tuned")
        self.fine_tuned_path = Config.getstr("general", "fine_tuned_path")
        self.__multi_gpu = Config.getboolean("general", "multi_gpu")
        self.__mode = Config.getstr("general", "mode")
        self.__pretrained_model_name_or_path = Config.getstr("model", "pretrained_model_name_or_path")
        self.__learning_rate = Config.getfloat("model", "learning_rate")
        self.__adam_epsilon = Config.getfloat("model", "adam_epsilon")
        self.__epochs = Config.getint("model", "epochs")
        self.__warmup_proportion = Config.getfloat("model", "warmup_proportion")
        self.__batch = Config.getint("model", "batch")
        self.__scheduler_type = Config.getstr("model", "scheduler_type")
        self.__learning_rate_decay_factor = Config.getfloat("model", "learning_rate_decay_factor")
        self.__patience = Config.getint("model", "patience")
        self.__weight_decay = Config.getfloat("model", "weight_decay")
        self.__grad_clipping = Config.getfloat("model", "grad_clipping", usefallback=True, fallback=1.0)

        self.__multi_task_training = MultiTaskDataset()
        self.__multi_task_validation = MultiTaskDataset()
        self.__multi_task_prediction = MultiTaskDataset()
        self.__tasks = {}
        self.__build(self.use_fine_tuned)

    def __addNER(self):
        ner = NER_encoder(self.bert_config)
        self.__tasks['ner'] = ner
        return ner

    def __addRE(self):
        re = RE_encoder(self.bert_config)
        self.__tasks['re'] = re
        return re

    def __to_device(self):

        if torch.cuda.is_available():
            self.__device = torch.device("cuda")
            self.__model = torch.nn.DataParallel(self.__model)
            self.cuda()
        else:
            self.__device = torch.device("cpu")

    def __get_pretrained(self, path):
        self.__model = BERT_EE_model.from_pretrained(path)
        self.__tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=Config.getboolean('model',
                                                                                               'tokenizer_lower_case'))

    def __build(self, use_fine_tuned):

        if use_fine_tuned:
            Logger.info('loading fine tuned model in ' + self.fine_tuned_path)
            self.__get_pretrained(self.fine_tuned_path)
        else:
            Logger.info('loading bert model in ' + self.__pretrained_model_name_or_path)
            self.__get_pretrained(self.__pretrained_model_name_or_path)

        self.bert_config = self.__model.bert.config
        tasks = [t.strip() for t in Config.getstr('general', 'tasks').split(',')]
        if len(self.__tasks) == 0 and len(tasks) > 0:

            load_test = False
            load_train = False
            load_validate = False

            if use_fine_tuned:
                load_test = self.__mode == 'predict'
                load_validate = self.__mode != 'predict'
            elif self.__mode == 'predict':
                load_test = load_train = load_validate = True
            else:
                load_train = load_validate = True

            if 'ner' in tasks:
                ner = self.__addNER()
                ner.build(self.__tokenizer, use_fine_tuned,
                          self.__mode == 'predict')
                if load_train:
                    self.__multi_task_training.add_task('ner', ner.training_dataset)
                if load_validate:
                    self.__multi_task_validation.add_task('ner', ner.validation_dataset)

            if 're' in tasks:
                re = self.__addRE()
                re.build(self.__tokenizer, use_fine_tuned,
                         self.__mode == 'predict')
                if load_train:
                    self.__multi_task_training.add_task('re', re.training_dataset)
                if load_validate:
                    self.__multi_task_validation.add_task('re', re.validation_dataset)
                if load_test:
                    if 'ner' not in tasks and Config.getstr('re', 'prediction_folder') == '':
                        Logger.error('Without NER prediction, RE prediction_folder must be be provided.')

            if load_train:
                self.__multi_task_training.build(self.__batch)
            if load_validate:
                self.__multi_task_validation.build(self.__batch)

        for t in self.__tasks.keys():
            self.__model.add_classifier(self.__tasks[t].classifier)
            self.__model.add_dropout(self.__tasks[t].dropout)

        params = list(self.__model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {
                'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.__learning_rate_decay_factor,  # 0.01,
                # 'lr': 2e-5,
                # 'ori_lr': 2e-5
            },
            {
                'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                # 'lr': 2e-5,
                # 'ori_lr': 2e-5
            }
        ]
        self.__optimizer = AdamW(grouped_params, correct_bias=False, lr=self.__learning_rate)

        if self.__scheduler_type == 'ms':
            milestones = [10, 20, 30]
            self.__scheduler = MultiStepLR(self.__optimizer, milestones=milestones,
                                           gamma=self.__learning_rate_decay_factor)
        elif self.__scheduler_type == 'exp':
            self.__scheduler = ExponentialLR(self.__optimizer, gamma=self.__learning_rate_decay_factor)
        elif self.__scheduler_type == 'rop':
            self.__scheduler = ReduceLROnPlateau(self.__optimizer, mode='max', factor=self.__learning_rate_decay_factor,
                                                 patience=3)
        elif self.__scheduler_type == 'linear':
            total_steps = self.__get_training_size() // self.__batch * self.__epochs
            warmup_steps = total_steps * self.__warmup_proportion
            self.__scheduler = get_linear_schedule_with_warmup(self.__optimizer,
                                                               num_warmup_steps=warmup_steps,
                                                               num_training_steps=total_steps)
        else:
            self.__scheduler = None

        self.__to_device()

    def __get_training_size(self):
        return self.__multi_task_training.__len__() * self.__batch

    def __get_num_training_labels(self, name):
        return len(self.__tasks[name].unique_labels)

    def __get_validation_dataset(self, name):
        return self.__tasks[name].validation_dataset

    def __get_training_dataset(self, name):
        return self.__tasks[name].training_dataset

    def __get_test_dataset(self, name):
        return self.__tasks[name].unlabelled_dataset

    def __start(self):
        for t in self.__tasks:
            self.__tasks[t].start()

    def __end(self):
        for t in self.__tasks:
            self.__tasks[t].end()

    def train_model(self):

        if not self.use_fine_tuned:
            self.__best_result = {}
            patience_count = 0
            dataset_size = len(self.__multi_task_training)
            indices = []

            for e in range(self.__epochs):

                self.__train_loader = DataLoader(self.__multi_task_training,
                                                 sampler=RandomSampler(self.__multi_task_training),
                                                 batch_size=1)
                Logger.info("**************{}**********".format('*' * str(e).__len__()))
                Logger.info("******** epoch {} ********".format(e))
                Logger.info("**************{}**********".format('*' * str(e).__len__()))

                self.train()
                self.__model.train()
                total_loss = 0
                for step, batch in enumerate(tqdm(self.__train_loader, desc="training")):

                    task_name = batch[1][0]

                    sample_idx, filenames, bert_labels, bert_tokens, tokens, tokens_indices, labels, logits, total_loss, loss = \
                        self.__update_step(task_name, self.__get_training_dataset(task_name),
                                           self.__multi_task_training.get_weights(task_name),
                                           batch[0], total_loss, predict=False)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.__model.parameters(), self.__grad_clipping)
                    self.__optimizer.step()
                    if self.__scheduler is not None:
                        self.__scheduler.step()
                    self.__optimizer.zero_grad()

                mean_loss = total_loss / len(self.__train_loader)
                Logger.info(" training loss : {0:.2f}".format(mean_loss))
                results = self.evaluate_model()
                if any([t not in self.__best_result for t in results]) or \
                        np.mean([results[t]['f1'] * self.__multi_task_training.get_weights(t) for t in self.__tasks]) > \
                        self.__best_result['overall_f1']:

                    for t in self.__tasks:
                        self.__tasks[t].save_best_results()

                    self.__best_result = results
                    self.__best_result['overall_f1'] = np.mean(
                        [results[t]['f1'] * self.__multi_task_training.get_weights(t) for t in results])
                    self.save()
                    patience_count = 0

                else:
                    patience_count += 1
                    Logger.info(" last {} epochs show no improvement".format(patience_count))
                    if patience_count == self.__patience:
                        Logger.info("*****************")
                        Logger.info("** Best result **")
                        Logger.info("*****************")
                        for t in self.__tasks:
                            self.__tasks[t].show_best_result()
                            self.load()

                        break

                Logger.info('>>>>> best:')
                if 'ner' in self.__best_result:
                    Logger.info('ner ', '{:.3f}'.format(self.__best_result['ner']['f1']))
                if 're' in self.__best_result:
                    Logger.info('re ', '{:.3f}'.format(self.__best_result['re']['f1']))
                if self.__best_result.keys().__len__() > 2:
                    Logger.info('avg ', '{:.3f}'.format(self.__best_result['overall_f1']))

                Logger.info('>>>>> current:')
                if 'ner' in results:
                    Logger.info('ner ', '{:.3f}'.format(results['ner']['f1']))
                if 're' in self.__best_result:
                    Logger.info('re ', '{:.3f}'.format(results['re']['f1']))
                if results.keys().__len__() > 2:
                    Logger.info('avg ', '{:.3f}'.format(
                        np.mean([results[t]['f1'] * self.__multi_task_training.get_weights(t) for t in self.__tasks])))

                if e == self.__epochs - 1:
                    Logger.info("*****************")
                    Logger.info("** Best result **")
                    Logger.info("*****************")
                    for t in self.__tasks:
                        self.__tasks[t].show_best_result()
                    self.load()


        elif self.__mode == 'train':
            Logger.info("evaluating fine tuned model")
            results = self.evaluate_model()


    def evaluate_model(self):

        self.__validation_loader = DataLoader(self.__multi_task_validation,
                                              sampler=RandomSampler(self.__multi_task_validation),
                                              batch_size=1)

        self.__start()
        self.eval()
        self.__model.eval()
        indices = []
        inputs = []
        labs = []
        prd = []
        with torch.no_grad():
            total_loss = 0
            for step, batch in enumerate(tqdm(self.__validation_loader, desc="evaluating")):
                task_name = batch[1][0]
                sample_idx, filenames, bert_labels, bert_tokens, tokens, tokens_indices, labels, logits, total_loss, loss = \
                    self.__update_step(task_name, self.__get_validation_dataset(task_name),
                                       self.__multi_task_validation.get_weights(task_name), batch[0], total_loss)

                preds = torch.argmax(logits, dim=-1)
                preds = preds.detach().cpu().numpy()
                bert_labels = bert_labels.detach().cpu().numpy()
                self.__tasks[task_name].update(sample_idx, filenames, bert_labels, preds, bert_tokens, tokens,
                                               tokens_indices, logits, labels)

            mean_loss = total_loss / len(self.__validation_loader)
            Logger.info(" evaluation loss : {0:.2f}".format(mean_loss))

        self.__end()
        results = {}
        for t in self.__tasks:
            results[t] = {}
            results[t]['acc'], results[t]['precsions'], results[t]['recalls'], results[t]['f1'] = self.__tasks[
                t].evaluate()
        return results

    def predict(self):
        if 'ner' in self.__tasks:
            for unlabelled_dataset in tqdm(self.__tasks['ner'].get_test_data(), desc="predicting NER",
                                           total=self.__tasks['ner'].get_test_data_total()):
                self.__multi_task_prediction = MultiTaskDataset()
                self.__multi_task_prediction.add_task('ner', unlabelled_dataset)
                self.__test_loader = DataLoader(self.__multi_task_prediction.build(1, randomize=False, multiple=False),
                                                sampler=SequentialSampler(self.__multi_task_prediction),
                                                batch_size=1)
                self.__do_predict('ner', unlabelled_dataset)
                self.generate_predictions('ner')

        if 're' in self.__tasks:
            if Config.getstr('re', 'prediction_folder') != '':
                unlabelled_folder = Config.getstr('re', 'prediction_folder')
            else:
                unlabelled_folder = Config.get_resultfolder('ner')

            for unlabelled_dataset in tqdm(self.__tasks['re'].get_test_data(unlabelled_folder), desc="predicting RE",
                                           total=self.__tasks['re'].get_test_data_total(unlabelled_folder)):

                if len(unlabelled_dataset) > 0:
                    self.__multi_task_prediction = MultiTaskDataset()
                    self.__multi_task_prediction.add_task('re', unlabelled_dataset)
                    self.__test_loader = DataLoader(
                        self.__multi_task_prediction.build(1, randomize=False, multiple=False),
                        sampler=SequentialSampler(self.__multi_task_prediction),
                        batch_size=1)
                    self.__do_predict('re', unlabelled_dataset)

                self.generate_predictions('re')

    def __do_predict(self, task, dataset):

        total_loss = 0
        self.__start()
        self.eval()
        self.__model.eval()
        labs = []
        prds = []
        with torch.no_grad():
            for step, batch in enumerate(self.__test_loader):
                task_name = batch[1][0]
                sample_idx, filenames, bert_labels, bert_tokens, tokens, tokens_indices, labels, logits, total_loss, loss = \
                    self.__update_step(task_name, dataset, self.__multi_task_prediction.get_weights(task_name),
                                       batch[0], total_loss, True)

                preds = torch.argmax(logits, dim=-1)
                preds = preds.detach().cpu().numpy()
                self.__tasks[task_name].update(sample_idx, filenames, bert_labels, preds, bert_tokens, tokens,
                                               tokens_indices, logits)
                labs.extend(bert_labels)
                prds.extend(preds)

        self.__end()

    def __update_step(self, task_name, dataset, task_weights, batch, total_loss, predict=False):

        batch_index = list(batch[-1].squeeze().detach().cpu().numpy()) if batch[-1].size()[1] > 1 else [
            int(batch[-1].squeeze().detach().cpu())]
        bert_tokens = [dataset.bert_tokens[i] for i in batch_index]
        tokens_indices = [dataset.token_indices[i] for i in batch_index]
        sample_idx = [dataset.sample_indices[i] for i in batch_index]
        filenames = [dataset.file_names[i] for i in batch_index]

        tokens = [dataset.tokens[i] for i in batch_index]
        labels = [dataset.labels[i] for i in batch_index]
        batch = tuple(p for p in batch[:-1])
        input_ids, token_type_ids, attention_mask, position_ids, bert_labels, masks = batch
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            bert_labels = bert_labels.cuda()

        input_ids = input_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        bert_labels = bert_labels.squeeze(0)

        outputs = self.__model(input_ids=input_ids, token_type_ids=None,
                               attention_mask=attention_mask,
                               position_ids=None, labels=bert_labels if not predict else None, head_mask=None,
                               task_id=list(self.__tasks.keys()).index(task_name),
                               num_labels=self.__get_num_training_labels(task_name))

        logits = outputs[0]
        loss = self.__tasks[task_name].criterion(bert_labels, attention_mask, logits)

        if torch.cuda.device_count() > 1 and self.__multi_gpu:
            loss = loss.mean()

        loss = loss * task_weights
        total_loss += loss.item()

        return sample_idx, filenames, bert_labels, bert_tokens, tokens, tokens_indices, labels, logits, total_loss, loss

    def save(self):

        model_path = Config.get_resultfolder('model')

        from transformers import WEIGHTS_NAME, CONFIG_NAME
        model_to_save = self.__model.module if torch.cuda.device_count() > 0 and self.__multi_gpu else self.__model
        torch.save(model_to_save.state_dict(), os.path.join(model_path, WEIGHTS_NAME))
        model_to_save.config.to_json_file(os.path.join(model_path, CONFIG_NAME))
        torch.save({'model_state_dict': model_to_save.state_dict()}, os.path.join(model_path, 'checkpoint'))
        self.__tokenizer.save_vocabulary(model_path)
        model_meta = {}
        for t in self.__tasks:
            model_meta[t] = self.__tasks[t].model_meta_data()

        json.dump(model_meta, open(os.path.join(model_path, 'model.json'), 'w'))

    def load(self, path=None):

        model_path = Config.get_resultfolder('model') if path is None else path
        model_to_load = self.__model.module if torch.cuda.device_count() > 0 and self.__multi_gpu else self.__model
        checkpoint = torch.load(os.path.join(model_path, 'checkpoint'),
                                map_location=torch.device('cpu') if not torch.cuda.is_available() else torch.device(
                                    'cuda'))
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

    def generate_predictions(self, task):
        self.__tasks[task].generate_predictions()
