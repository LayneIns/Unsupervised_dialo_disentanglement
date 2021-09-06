import sys 
import os 
import argparse
from tqdm import tqdm 
import json 
from time import strftime, gmtime
import logging
import numpy as np
import random

import torch
import transformers 
from transformers import BertTokenizer
import utils
from model import MatchModel
from trainer import supervised_trainer


def preprare(args):
    current_time = strftime("%Y-%b-%d-%H_%M_%S", gmtime())
    args.current_time = current_time

    args.data_path = os.path.join(args.data_dir, args.data_name, 'tokenized', "{}.json".format(args.data_type))
    args.reward_path = os.path.join(args.reward_path, str(args.round), "reward", "reward.pt")

    args.output_dir = os.path.join(args.output_dir, args.data_name, str(args.round), args.current_time)
    args.cache_dir = os.path.join(args.output_dir, args.cache_dir)
    args.log_file = os.path.join(args.output_dir, 'log.txt')
    os.makedirs(args.cache_dir, exist_ok=True)

    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=args.log_file, filemode='a')
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
    parser.add_argument('--data_dir', type=str, default="../../../data/")
    parser.add_argument('--data_name', type=str, default="movie")
    parser.add_argument('--data_type', type=str, default="train")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--cache_dir', type=str, default="cached")
    parser.add_argument('--init_checkpoint', type=str, required=True)
    parser.add_argument('--word_dict', type=str, required=True)
    parser.add_argument('--reward_path', type=str, default="../bert-reward/output/")
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--seq_max_len', type=int, default=100, \
                        help="the maximum length of a sequence")
    parser.add_argument('--min_token_freq', type=int, default=5, \
                        help="the minimum frequency of a token")
    parser.add_argument('--max_session_number', type=int, default=4)
    parser.add_argument('--per_gpu_batch_size', type=int, default=32)
    parser.add_argument('--num_warmup_steps', type=int, default=30)
    parser.add_argument('--coherency_reward_weight', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--emb_size', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument("--bidirectional", action='store_true',
                        help="Whether not to use bidirectional encoder")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--rnn_cell', type=str, choices=['gru', 'lstm'], default="lstm")
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=100)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    logger, tokenizer = preprare(args)
    set_random_seed(args)

    raw_data = utils.read_raw_data(args.data_path)
    logger.info("{} data instances read from the original dataset.".format(len(raw_data)))

    tokenized_data = utils.tokenize_data(raw_data, tokenizer)
    logger.info("{} tokenized data read.".format(len(tokenized_data)))

    word_dict = torch.load(args.word_dict)
    args.pad_id = word_dict['<PAD>']

    word_emb_matrix = None

    training_data_id = utils.convert_data_to_id(tokenized_data, word_dict)
    logger.info("{} data id read.".format(len(training_data_id)))

    args.max_context_len = utils.get_max_context_length(training_data_id)
    logger.info("Max context length: {}".format(args.max_context_len))

    reward = torch.load(args.reward_path)
    coherency_reward, speaker_reward = utils.parse_reward(reward, tokenized_data)

    model = MatchModel(args.rnn_cell, len(word_dict), word_emb_matrix, 
                    args.hidden_size, args.emb_size, args.max_context_len, 
                    bidirectional=args.bidirectional)

    supervised_trainer(args, model, training_data_id, coherency_reward, speaker_reward, logger)

if __name__ == "__main__":
    main()