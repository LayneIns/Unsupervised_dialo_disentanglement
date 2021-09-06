import sys 
import os 
import argparse
from tqdm import tqdm 
import json 
from time import strftime, gmtime
import logging
import numpy as np
import random
import glob
import re
from sklearn.metrics import f1_score, recall_score, precision_score

import torch
import transformers 
from transformers import BertTokenizer
import utils
from model import MatchModel
from trainer import supervised_trainer
from data_loader import DatasetforMM
from torch.utils.data import DataLoader


def write_result(out_labels, preds, args):
    label = out_labels.tolist()
    preds = preds.tolist()
    results = []
    for i in range(len(preds)):
        one_pred = preds[i]
        one_label = label[i]
        one_item = {
            'pred': one_pred,
            'label': one_label
        }
        results.append(one_item)
    checkpoint_num = re.findall(r"ckpt-(.+?)\.pkl", args.checkpoint)[0]
    filename = os.path.join(args.result_dir, "{}_checkpoint_{}.json".format(args.data_type, checkpoint_num))
    with open(filename, 'w') as fout:
        json.dump(results, fout, indent=2)


def preprare(args):
    args.data_path = os.path.join(args.data_dir, args.data_name, args.data_mode, "{}.json".format(args.data_type))

    args.result_dir = os.path.join(args.output_dir, args.result_dir)
    args.cache_dir = os.path.join(args.output_dir, args.cache_dir)
    args.cached_input_dir = os.path.join(args.cached_input_dir, args.data_name, args.data_mode)
    args.log_file = os.path.join(args.output_dir, '{}_log.txt'.format(args.data_type))
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.cached_input_dir, exist_ok=True)

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
    parser.add_argument('--data_dir', type=str, default="../data/")
    parser.add_argument('--data_name', type=str, default="movie")
    parser.add_argument('--data_mode', type=str, default="single")
    parser.add_argument('--data_type', type=str, choices=['test', 'dev'], default="dev")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--result_dir', type=str, default="result/")
    parser.add_argument('--cache_dir', type=str, default="cached/")
    parser.add_argument('--cached_input_dir', type=str, default="cached_input")
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--seq_max_len', type=int, default=100, \
                        help="the maximum length of a sequence")
    parser.add_argument('--per_gpu_batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--emb_size', type=int, default=300)
    parser.add_argument("--bidirectional", action='store_true',
                        help="Whether not to use bidirectional encoder")
    parser.add_argument("--read_cached_input", action='store_true',
                        help="Whether not to read cached input")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--rnn_cell', type=str, choices=['gru', 'lstm'], default="lstm")
    parser.add_argument('--random_seed', type=int, default=1234)
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
    else:
        tokenized_data = None
    
    word_dict_filename = os.path.join(args.cached_input_dir, 'word_dict.pt')
    word_dict = torch.load(word_dict_filename)
    args.pad_id = word_dict['<PAD>']

    word_emb_matrix = None

    data_id_filename = os.path.join(args.cached_input_dir, '{}_data_id.pt'.format(args.data_type))
    data_id = utils.convert_data_to_id(tokenized_data, word_dict, data_id_filename, logger)

    # args.max_context_len = utils.get_max_context_length(training_data_id, logger)
    # logger.info("Max context length: {}".format(args.max_context_len))
    args.max_context_len = -1

    if args.checkpoint == -1:
        checkpoints = list(sorted(glob.glob(args.cache_dir + "ckpt-*", recursive=True)))
    else:
        checkpoints = [os.path.join(args.cache_dir, "ckpt-{}.pkl".format(args.checkpoint))]
    
    print(checkpoints)

    for checkpoint in tqdm(checkpoints):
        args.checkpoint = checkpoint
        logger.info("{} results for checkpoint: {}".format(args.data_type, args.checkpoint))
        model = MatchModel(args.rnn_cell, len(word_dict), word_emb_matrix, 
                    args.hidden_size, args.emb_size, args.max_context_len, 
                    bidirectional=args.bidirectional)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(checkpoint))

        model.to(args.device)

        dev_dataset = DatasetforMM(data_id, args.seq_max_len, len(data_id), args.pad_id)
        # Test!
        logger.info("  ***** Running Testing *****  *")
        logger.info("  Num examples = %d", len(dev_dataset))
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)

        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=args.per_gpu_batch_size,
            collate_fn=utils.batch_list_to_batch_tensors)

        dev_iterator = tqdm(
            dev_dataloader, initial=0,
                desc="Iter (loss=X.XXX, lr=X.XXXXXXX)")
        
        preds = None

        model.eval()
        for batch in dev_iterator:
            batch = tuple(t.to(args.device) for t in batch)
            context, context_seq_len, context_len, current_message, current_message_seq_len, label = batch
            outputs = model(context, context_seq_len, context_len, current_message, current_message_seq_len)
            prob = outputs
            
            if preds is None:
                preds = prob.detach().cpu().numpy()
                out_labels = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, prob.detach().cpu().numpy(), axis=0)
                out_labels = np.append(out_labels, label.detach().cpu().numpy(), axis=0) 
        preds_arg = np.argmax(preds, axis=1)
        write_result(out_labels, preds, args)
        precision = precision_score(out_labels, preds_arg)
        recall = recall_score(out_labels, preds_arg)
        f1 = f1_score(out_labels, preds_arg)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        logger.info("{} result for checkpoint {}".format(args.data_type, args.checkpoint))
        result_msg = "precision: {}, recall : {}, f1: {}".format(precision, recall, f1)
        logger.info(result_msg)
        print(result_msg)


if __name__ == "__main__":
    main()
