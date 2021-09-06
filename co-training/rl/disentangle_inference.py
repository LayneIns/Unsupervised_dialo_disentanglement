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
import itertools
import re
from sklearn.metrics import f1_score, recall_score, precision_score

import torch
import transformers 
from transformers import BertTokenizer
from process1.model import MatchModel
# from process1.trainer import supervised_trainer
from torch.utils.data import DataLoader


class DatasetforDisentanglement(torch.utils.data.Dataset):
    def __init__(self, training_data, seq_max_len, num_training_instances, pad_id):
        self.training_data = training_data
        self.seq_max_len = seq_max_len
        self.num_training_instances = num_training_instances
        self.pad_id = pad_id
    
    def __len__(self):
        return self.num_training_instances
    
    def __trunk(self, ids, max_len):
        if len(ids) > max_len:
            ids = ids[:max_len]
        return ids
    
    def __pad(self, ids, max_len):
        if len(ids) < max_len:
            return ids + [self.pad_id] * (max_len - len(ids))
        else:
            assert len(ids) == max_len
            return ids
    
    def __getitem__(self, idx):
        idx = idx % len(self.training_data)
        example = self.training_data[idx]
        messages = []
        labels = []
        message_seq_len = []
        speakers = []
        for item in example:
            text = item['message']
            label = item['label']
            speaker = item['speaker']
            text = self.__trunk(text, self.seq_max_len)
            text_seq_len = len(text)
            text = self.__pad(text, self.seq_max_len)
            messages.append(text)
            message_seq_len.append(text_seq_len)
            labels.append(label)
            speakers.append(speaker)
        return messages, message_seq_len, labels, len(messages), speakers


def batch_list_to_batch_tensors(batch):
    new_batch = list(zip(*batch))
    context, context_seq_len, label_id, context_len, speakers = new_batch

    context = list(itertools.chain(*context))
    context_tensor = torch.tensor(context, dtype=torch.long)
    context_seq_len = list(itertools.chain(*context_seq_len))
    context_seq_len_tensor = torch.tensor(context_seq_len, dtype=torch.int64)

    context_len = torch.tensor(context_len, dtype=torch.long)

    return context_tensor, context_seq_len_tensor, context_len, label_id, speakers


def read_raw_data(data_path):
    print("Reading original data ...")
    with open(data_path) as fin:
        data = json.load(fin)
    print("{} data instances read from the original dataset.".format(len(data)))
    return data


def tokenize_data(data, cached_features_file, tokenizer, logger):
    print("Tokenizing data ...")
    tokenized_data = []
    for item in tqdm(data):
        one_dialogue = []
        for message in item:
            speaker = message['speaker']
            text = message['text'].split()
            label = message['label']
            one_dialogue.append({
                'speaker': speaker,
                'message': text,
                'label': label
            })
        tokenized_data.append(one_dialogue)
    print("Saving tokenized data into cached file {}".format(cached_features_file))
    
    logger.info("{} tokenized data read.".format(len(tokenized_data)))
    return tokenized_data


def convert_data_to_id(data, word_dict, cached_features_file, logger):
    data_id = []
    for item in tqdm(data):
        new_item = []
        for message in item:
            text = message['message']
            label = message['label']
            speaker = message['speaker']
            message_id = []
            for token in text:
                message_id.append(word_dict.get(token, word_dict.get('<UNK>')))
            new_item.append({
                'message': message_id,
                'label': label,
                'speaker': speaker
            })
        data_id.append(new_item)
    print("Saving data id into cached file {}".format(cached_features_file))

    logger.info("{} data id read.".format(len(data_id)))
    return data_id


def get_max_context_length(data_id, logger):
    print("Retrieving the maximum context length ...")
    context_length = 0
    for item in tqdm(data_id):
        context_length = max(len(item), context_length)
    logger.info("Max context length: {}".format(context_length))
    return context_length


def write_result(labels, preds, args):
    results = {}
    for i in range(len(labels)):
        one_pred = preds[i]
        one_label = labels[i]
        one_item = {
            'pred': one_pred,
            'label': one_label
        }
        results[i] = one_item
    checkpoint_num = re.findall(r"ckpt-(.+?)\.pkl", args.checkpoint)[0]
    filename = os.path.join(args.result_dir, "{}_checkpoint_{}.json".format(args.data_type, checkpoint_num))
    with open(filename, 'w') as fout:
        json.dump(results, fout, indent=2)


def preprare(args):
    args.data_path = os.path.join(args.data_dir, args.data_name, 'tokenized', "{}.json".format(args.data_type))
    args.cache_dir = os.path.join(args.output_dir, args.cache_dir)

    args.output_dir = os.path.join(args.output_dir, "disentanglement")
    os.makedirs(args.output_dir, exist_ok=True)
    args.result_dir = os.path.join(args.output_dir, args.result_dir)
    
    args.cached_input_dir = os.path.join(args.model_type, args.cached_input_dir, args.data_name, "disentanglement")
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
    parser.add_argument('--data_dir', type=str, default="../../data/")
    parser.add_argument('--data_name', type=str, default="movie")
    parser.add_argument('--data_type', type=str, choices=['test', 'dev', 'train'], default="dev")
    parser.add_argument('--model_type', type=str, default="process1")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--result_dir', type=str, default="disentangle_result/")
    parser.add_argument('--cache_dir', type=str, default="cached/")
    parser.add_argument('--cached_input_dir', type=str, default="cached_input")
    parser.add_argument('--word_dict', type=str, default="./process1/cached_input/movie/single/word_dict.pt")
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--seq_max_len', type=int, default=100, \
                        help="the maximum length of a sequence")
    parser.add_argument('--per_gpu_batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--emb_size', type=int, default=300)
    parser.add_argument('--max_session_number', type=int, default=4)
    parser.add_argument("--bidirectional", action='store_true',
                        help="Whether not to use bidirectional encoder")
    parser.add_argument("--read_cached_input", action='store_true',
                        help="Whether not to read cached input")
    # parser.add_argument("--multiple_gpu", action='store_true',
    #                     help="Whether not to use multiple gpu")
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

    raw_data = read_raw_data(args.data_path)
    # print items for debug
    for item in raw_data[0:2]:
        print(item)
    # end debug

    tokenized_data_filename = os.path.join(args.cached_input_dir, 'tokenized_{}.pt'.format(args.data_type))
    tokenized_data = tokenize_data(raw_data, tokenized_data_filename, tokenizer, logger)

    word_dict = torch.load(args.word_dict)
    args.pad_id = word_dict['<PAD>']

    word_emb_matrix = None

    data_id_filename = os.path.join(args.cached_input_dir, '{}_data_id.pt'.format(args.data_type))
    data_id = convert_data_to_id(tokenized_data, word_dict, data_id_filename, logger)

    args.max_context_len = get_max_context_length(data_id, logger)
    print("Max context length: {}".format(args.max_context_len))

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
                    bidirectional=args.bidirectional, max_session_number=args.max_session_number, 
                    inference=True)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(checkpoint))

        model.to(args.device)

        dev_dataset = DatasetforDisentanglement(data_id, args.seq_max_len, len(data_id), args.pad_id)

        # Test!
        logger.info("  ***** Running Testing *****  *")
        logger.info("  Num examples = %d", len(dev_dataset))
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)

        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=args.per_gpu_batch_size,
            collate_fn=batch_list_to_batch_tensors)

        dev_iterator = tqdm(
            dev_dataloader, initial=0,
                desc="Iter (loss=X.XXX, lr=X.XXXXXXX)")

        preds = None

        model.eval()
        for batch in dev_iterator:
            context, context_seq_len, context_len, label, speakers = batch
            context = context.to(args.device)
            context_len = context_len.to(args.device)
            max_context_len = max(context_len)
            context_vector = model(context, context_seq_len, context_len)
            context_vector = model.module.build_context_batch(context_vector, context_len, max_context_len)

            batch_size = context_vector.size()[0]

            decision_sequence = torch.zeros([batch_size, 1], dtype=torch.int64, device=args.device)
            session_number = torch.ones([batch_size, 1], dtype=torch.int64, device=args.device)
            for step in range(1, max_context_len):
                current_context = context_vector[:, 0:step, :]
                current_message = context_vector[:, step, :]
                values, new_session_value = model.module.get_decision_v2(current_context, current_message, decision_sequence, session_number)
                current_decision = []
                for i in range(batch_size):
                    max_num, max_value_index = values[i]
                    new_session_score = new_session_value[i]
                    speaker_list = speakers[i]
                    flag = False
                    if step < len(speaker_list):
                        speaker = speaker_list[step]
                        speaker_index = speaker_list[:step].index(speaker) if speaker in speaker_list[:step] else -1
                        if speaker_index != -1:
                            flag = True
                            one_decision = decision_sequence[i][speaker_index]
                    if flag:
                        current_decision.append(max_value_index)
                    else:
                        if new_session_score < 0:
                            if session_number[i] < args.max_session_number:
                                current_decision.append(session_number[i].data.item())
                                session_number[i] += 1
                            else:
                                current_decision.append(max_value_index)
                        else:
                            current_decision.append(max_value_index)
                current_decision = torch.tensor(current_decision, dtype=torch.int64, device=args.device).view(batch_size, 1)
                decision_sequence = torch.cat([decision_sequence, current_decision], dim=1)
            if preds is None:
                preds = decision_sequence.detach().cpu().numpy().tolist()
                preds = [preds[i][:context_len[i]] for i in range(batch_size)]
                out_labels = label
            else:
                one_preds = decision_sequence.detach().cpu().numpy().tolist()
                one_preds = [one_preds[i][:context_len[i]] for i in range(batch_size)]
                preds += one_preds
                out_labels += label
        write_result(out_labels, preds, args)


if __name__ == "__main__":
    main()