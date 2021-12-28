import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
from config import Config
from logger import Logger
from util import *
from model import BERT_EE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--general_seed', type=int, default=None)
    parser.add_argument('--general_tasks', default=None)
    parser.add_argument('--general_log_level', default=None)
    parser.add_argument('--general_multi_gpu', action='store_true', default=None)
    parser.add_argument('--general_mode', default=None)
    parser.add_argument('--general_results_path', default=None)
    parser.add_argument('--general_results_folder', default=None)
    parser.add_argument('--general_fine_tuned_path', default=None)
    parser.add_argument('--general_use_fine_tuned', action='store_true', default=None)
    parser.add_argument('--general_use_sent_segmentation', action='store_true', default=None)
    parser.add_argument('--general_use_spacy_tokenizer', action='store_true', default=None)
    parser.add_argument('--general_spacy_model',default=None)
    parser.add_argument('--ner_labels', default=None)
    parser.add_argument('--ner_scheme', default=None)
    parser.add_argument('--ner_training_folder', default=None)
    parser.add_argument('--ner_validation_folder', default=None)
    parser.add_argument('--ner_prediction_folder', default=None)
    parser.add_argument('--ner_dropout', type=float, default=None)
    parser.add_argument('--ner_eval_mode',  default=None)
    parser.add_argument('--re_labels', default=None)
    parser.add_argument('--re_training_folder', default=None)
    parser.add_argument('--re_validation_folder', default=None)
    parser.add_argument('--re_prediction_folder', default=None)
    parser.add_argument('--re_dropout', type=float, default=None)
    parser.add_argument('--re_distance', type=int, default=None)
    parser.add_argument('--re_no_relation_ratio', type=float, default=None)
    parser.add_argument('--re_num_workers', type=int, default=None)
    parser.add_argument('--re_max_sent_windows', type=int, default=None)
    parser.add_argument('--model_max_seq_len', type=int, default=None)
    parser.add_argument('--model_pretrained_model_name_or_path', default=None)
    parser.add_argument('--model_tokenizer_lower_case', action='store_true', default=None)
    parser.add_argument('--model_batch', type=int, default=None)
    parser.add_argument('--model_epochs', type=int, default=None)
    parser.add_argument('--model_patience', type=int, default=None)
    parser.add_argument('--model_grad_clipping', type=float, default=None)
    parser.add_argument('--model_learning_rate', type=float, default=None)
    parser.add_argument('--model_weight_decay', type=float, default=None)
    parser.add_argument('--model_adam_epsilon', type=float, default=None)
    parser.add_argument('--model_warmup_proportion', type=float, default=None)
    parser.add_argument('--model_warmup_step', type=int, default=None)
    parser.add_argument('--model_scheduler_type', default=None)
    parser.add_argument('--model_learning_rate_decay_factor', type=float, default=None)

    args = parser.parse_args()

    Config.init(args)
    Logger.init()
    command_params = ','.join(['{}={} '.format(k,p) for k,p in dict(Config.args).items() if p is not None])
    if len(command_params)>0:
        Logger.info('Command params: {}'.format(command_params))

    set_seed(Config.getint("general", "seed"))
    model = BERT_EE()

    model.train_model()

    if Config.getstr('general','mode').lower() == 'predict':
        model.predict()