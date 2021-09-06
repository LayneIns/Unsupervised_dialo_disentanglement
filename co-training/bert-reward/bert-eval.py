import os 
import sys 
import argparse
import json 
import re
from tqdm import tqdm 
import numpy as np 
from time import strftime, gmtime
import logging

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.functional import softmax
import transformers 
from transformers import BertModel, BertTokenizer
import apex 


class DatasetForBertEval(torch.utils.data.Dataset):
    def __init__(self, features, max_source_len, cls_id, sep_id, pad_id, mask_id):
        self.features = features
        self.max_source_len = max_source_len
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.mask_id = mask_id
    
    def __len__(self):
        return len(self.features)
    
    def __trunk(self, ids, max_len):
        if len(ids) > max_len - 1:
            ids = ids[:max_len - 1]
        ids = ids + [self.sep_id]
        return ids

    def __pad(self, ids, max_len):
        if len(ids) < max_len:
            return ids + [self.pad_id] * (max_len - len(ids))
        else:
            assert len(ids) == max_len
            return ids
    
    def __getitem__(self, idx):
        feature = self.features[idx]

        source_id_1 = self.__trunk([self.cls_id] + feature["text1"], self.max_source_len)
        segment_id_1 = [0]*len(source_id_1) + [0]*(self.max_source_len-len(source_id_1))
        input_mask_1 = [1]*len(source_id_1) + [0]*(self.max_source_len-len(source_id_1))
        input_id_1 = self.__pad(source_id_1, self.max_source_len)

        source_id_2 = self.__trunk([self.cls_id] + feature["text2"], self.max_source_len)
        segment_id_2 = [0]*len(source_id_2) + [0]*(self.max_source_len-len(source_id_2))
        input_mask_2 = [1]*len(source_id_2) + [0]*(self.max_source_len-len(source_id_2))
        input_id_2 = self.__pad(source_id_2, self.max_source_len)

        pos1 = feature['pos1']
        pos2 = feature['pos2']
        label1 = feature['label1']
        label2 = feature['label2']
        data_id = feature['data_id']
        return input_id_1, input_mask_1, segment_id_1, \
                    input_id_2, input_mask_2, segment_id_2, \
                    pos1, pos2, label1, label2, data_id


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def read_data(filename):
    print("Reading data from {} ...".format(filename))
    with open(filename) as fin:
        data = json.load(fin)
    print("{} original data read".format(len(data)))
    return data


def convert_data_to_id(data, tokenizer, filename):
    if os.path.exists(filename):
        print("Loading tokenized data ...")
        new_data = torch.load(filename)
        print("{} data loaded".format(len(new_data)))
    else:
        print("Converting data text to id ...")
        new_data = []
        for item in tqdm(data):
            text1 = item['text1']
            text2 = item['text2']
            text_id1 = tokenizer.convert_tokens_to_ids(text1.split())
            text_id2 = tokenizer.convert_tokens_to_ids(text2.split())
            new_data.append({
                'text1': text_id1,
                'label1': item['label1'],
                'pos1': item['pos1'],
                'text2': text_id2,
                'label2': item['label2'],
                'pos2': item['pos2'],
                'data_id': item['data_id'],
            })
        print("{} data converted".format(len(new_data)))
        torch.save(new_data, filename)
    return new_data    
        

def evaluate(args, model, tokenizer, data, logger):
    if args.fp16:
        from apex import amp 
    else:
        amp = None
    
    if amp:
        model = amp.initialize(model, opt_level=args.fp16_opt_level)
    
    eval_dataset = DatasetForBertEval(
        features=data, max_source_len=args.max_source_len,
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, 
        pad_id=tokenizer.pad_token_id, mask_id=tokenizer.mask_token_id
    )
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, 
                                batch_size=args.eval_batch_size,
                                collate_fn=batch_list_to_batch_tensors)
    
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Num of GPUs = %d", args.n_gpu)
    all_pos1 = None
    all_pos2 = None
    all_label1 = None
    all_label2 = None
    all_data_ids = None
    all_probs = None

    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = [t.to(args.device) for t in batch]
            input_id_1, input_mask_1, segment_id_1, \
                input_id_2, input_mask_2, segment_id_2, \
                pos1, pos2, label1, label2, data_id = batch
            inputs_1 = {'input_ids': input_id_1,
                        'attention_mask': input_mask_1,
                        }
            if args.model_config.startswith('bert'):
                inputs_1['token_type_ids'] = segment_id_1
            sent1_vec = model(**inputs_1)[1]
            
            inputs_2 = {'input_ids': input_id_2,
                        'attention_mask': input_mask_2,
                        }
            if args.model_config.startswith('bert'):
                inputs_2['token_type_ids'] = segment_id_2
            sent2_vec = model(**inputs_2)[1]

            score = torch.matmul(sent1_vec.unsqueeze(1), sent2_vec.unsqueeze(1).permute(0, 2, 1)).squeeze(-1).squeeze(-1)
            probs = nn.Sigmoid()(score)
            if all_probs is None:
                all_probs = probs.detach().cpu().numpy()
                all_pos1 = pos1.detach().cpu().numpy()
                all_pos2 = pos2.detach().cpu().numpy()
                all_label1 = label1.detach().cpu().numpy()
                all_label2 = label2.detach().cpu().numpy()
                all_data_ids = data_id.detach().cpu().numpy()
            else:
                all_probs = np.append(all_probs, probs.detach().cpu().numpy(), axis=0)
                all_pos1 = np.append(all_pos1, pos1.detach().cpu().numpy(), axis=0)
                all_pos2 = np.append(all_pos2, pos2.detach().cpu().numpy(), axis=0)
                all_label1 = np.append(all_label1, label1.detach().cpu().numpy(), axis=0)
                all_label2 = np.append(all_label2, label2.detach().cpu().numpy(), axis=0)
                all_data_ids = np.append(all_data_ids, data_id.detach().cpu().numpy(), axis=0)
    
    write_results(args, all_probs, all_pos1, all_pos2, all_label1, all_label2, all_data_ids)


def write_results(args, probs, pos1, pos2, label1, label2, data_ids):
    def get_data_length(pos1, pos2, data_ids):
        lengths = {}
        for i in range(len(data_ids)):
            if data_ids[i] not in lengths:
                lengths[data_ids[i]] = 0
        for i in range(len(data_ids)):
            lengths[data_ids[i]] = max(pos1[i]+1, lengths[data_ids[i]])
            lengths[data_ids[i]] = max(pos2[i]+1, lengths[data_ids[i]])
        return lengths
    
    temp_filename = os.path.join(args.result_dir, 'tmp.pt')
    torch.save([probs, pos1, pos2, label1, label2, data_ids], temp_filename)

    reward_filename = os.path.join(args.result_dir, "reward.pt")

    reward = {}
    data_lengths = get_data_length(pos1, pos2, data_ids)

    for key, val in data_lengths.items():
        reward[key] = {
            'reward': np.ones([val, val], dtype=np.float64),
            'label': np.empty([val], dtype=np.int32)
        }
    
    for i in range(len(data_ids)):
        one_data_id = data_ids[i]
        one_prob = probs[i]
        one_pos1 = pos1[i]
        one_pos2 = pos2[i]
        one_label1 =label1[i]
        one_label2 =label2[i]
        reward[one_data_id]['reward'][one_pos1][one_pos2] = one_prob
        reward[one_data_id]['reward'][one_pos2][one_pos1] = one_prob
        reward[one_data_id]['label'][one_pos1] = one_label1
        reward[one_data_id]['label'][one_pos2] = one_label2
    torch.save(reward, reward_filename)


def prepare(args):
    args.output_dir = os.path.join(args.output_dir, args.round)
    os.makedirs(args.output_dir, exist_ok=True)

    args.result_dir = os.path.join(args.output_dir, args.result_dir)
    os.makedirs(args.result_dir, exist_ok=True)

    args.cached_input_dir = os.path.join(args.cached_input_dir, args.data_name)
    os.makedirs(args.cached_input_dir, exist_ok=True)
    
    # define the logger
    logger_name = os.path.join(args.output_dir, "log.txt")
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logger_name, filemode='a')
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    logger_msg = '\n\n========================='
    logger.info(logger_msg)
    logger.info(json.dumps(args.__dict__, sort_keys=True, indent=4))
    args.device = device

    if args.fp16:
        try:
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    return logger


def get_model_and_tokenizer(args):
    if args.model_path is not 'none':
        model = BertModel.from_pretrained(args.model_path)
    else:
        model = BertModel.from_pretrained(args.model_config)
    tokenizer = BertTokenizer.from_pretrained(args.model_config)
    return model, tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--data_name', type=str, choices=['movie'], default='movie')
    parser.add_argument('--model_config', type=str, default='bert-base-uncased')
    parser.add_argument('--model_path', type=str, default='none')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--result_dir', type=str, default='reward')
    parser.add_argument('--cached_input_dir', type=str, default='cached_input')
    parser.add_argument('--max_source_len', type=int, default=100)
    parser.add_argument('--round', type=str, default='1')
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=32)
    parser.add_argument('--read_cached_input', action='store_true', 
                        help="Read cached input or not")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logger = prepare(args)

    model, tokenizer = get_model_and_tokenizer(args)
    
    if not args.read_cached_input:
        filename = os.path.join(args.data_dir, args.data_name, '{}.json'.format(args.data_type))
        raw_data = read_data(filename)
        logger.info("{} original data read".format(len(raw_data)))
    else:
        raw_data = None

    cached_tokenized_filename = os.path.join(args.cached_input_dir, "{}_data_id.pt".format(args.data_type))
    data_id = convert_data_to_id(raw_data, tokenizer, cached_tokenized_filename)
    logger.info("{} data converted".format(len(data_id)))
    
    # cached_flat_data_filename = os.path.join(args.cached_input_dir, "flat_tokenized_{}.pt".format(args.data_type)) 
    # data, labels = flatten_utterance(tokenized_data, tokenizer, cached_flat_data_filename)

    model.to(args.device)

    # res_filename = os.path.join(args.result_dir, '{}_res.pk'.format(args.data_type))
    evaluate(args, model, tokenizer, data_id, logger)
    


if __name__ == "__main__":
    main()

