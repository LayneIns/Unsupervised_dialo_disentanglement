import sys 
import os 
import argparse
from tqdm import tqdm 
import json 
from time import strftime, gmtime
import logging
import numpy as np
import random
import gc 

import torch
import transformers 
from transformers import BertTokenizer
import utils
from model import MatchModel
from trainer import supervised_trainer


def preprare(args):
    current_time = strftime("%Y-%b-%d-%H_%M_%S", gmtime())
    args.current_time = current_time

    args.data_path = os.path.join(args.data_dir, args.data_name, args.data_mode, "{}.json".format(args.data_type))

    args.output_dir = os.path.join(args.output_dir, args.data_name, args.data_mode, args.current_time)
    args.cache_dir = os.path.join(args.output_dir, args.cache_dir)
    args.cached_input_dir = os.path.join(args.cached_input_dir, args.data_name, args.data_mode)
    args.log_file = os.path.join(args.output_dir, 'log.txt')
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.cached_input_dir, exist_ok=True)

    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=args.log_file, filemode='w')
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    logger.info(json.dumps(args.__dict__, sort_keys=True, indent=4))
    args.device = device

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return logger, tokenizer


def set_random_seed(args):
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.random_seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/")
    parser.add_argument('--data_name', type=str, default="movie")
    parser.add_argument('--data_mode', type=str, default="single")
    parser.add_argument('--data_type', type=str, default="train")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--cache_dir', type=str, default="cached")
    parser.add_argument('--glove_path', type=str, default='../../../../files/glove/glove.840B.300d.txt')
    parser.add_argument('--cached_input_dir', type=str, default="cached_input")
    parser.add_argument('--seq_max_len', type=int, default=100, \
                        help="the maximum length of a sequence")
    parser.add_argument('--min_token_freq', type=int, default=5, \
                        help="the minimum frequency of a token")
    parser.add_argument('--per_gpu_batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--emb_size', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--bidirectional", action='store_true',
                        help="Whether not to use bidirectional encoder")
    parser.add_argument("--read_cached_input", action='store_true',
                        help="Whether not to read cached input")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--rnn_cell', type=str, choices=['gru', 'lstm'], default="lstm")
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--logging_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=5000)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    logger, tokenizer = preprare(args)
    set_random_seed(args)

    if not args.read_cached_input:
        raw_data = utils.read_raw_data(args.data_path)
        # print items for debug
        for item in raw_data[0:2]:
            print(item)
        # end debug
        tokenized_data_filename = os.path.join(args.cached_input_dir, 'tokenized_{}.pt'.format(args.data_type))
        tokenized_data = utils.tokenize_data(raw_data, tokenized_data_filename, tokenizer)
        logger.info("{} tokenized data read.".format(len(tokenized_data)))
        del raw_data
        gc.collect()
    else:
        tokenized_data = None
    

    word_dict_filename = os.path.join(args.cached_input_dir, 'word_dict.pt')
    word_dict = utils.get_word_dict(tokenized_data, word_dict_filename, args.min_token_freq, logger)
    args.pad_id = word_dict['<PAD>']

    word_emb_matrix_filename = os.path.join(args.cached_input_dir, 'cached_word_emb_freq_{}.pt'.format(str(args.min_token_freq)))
    word_emb_matrix = utils.build_word_emb_matrix(word_dict, args.glove_path, args.emb_size, logger, word_emb_matrix_filename)

    data_id_filename = os.path.join(args.cached_input_dir, '{}_data_id.pt'.format(args.data_type))
    training_data_id = utils.convert_data_to_id(tokenized_data, word_dict, data_id_filename, logger)

    del tokenized_data
    gc.collect()

    args.max_context_len = utils.get_max_context_length(training_data_id, logger)
    logger.info("Max context length: {}".format(args.max_context_len))

    model = MatchModel(args.rnn_cell, len(word_dict), word_emb_matrix, 
                    args.hidden_size, args.emb_size, args.max_context_len, 
                    bidirectional=args.bidirectional)
    
    supervised_trainer(args, model, training_data_id, logger)

if __name__ == "__main__":
    main()