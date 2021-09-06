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
    def __init__(self, features, max_source_len, max_len, cls_id, sep_id, pad_id, mask_id):
        self.features = features
        self.max_source_len = max_source_len
        self.max_seq_len = max_len
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

        source_id_1 = self.__trunk([self.cls_id] + feature["text1_id"], self.max_source_len)
        segment_id_1 = [0]*len(source_id_1) + [0]*(self.max_source_len-len(source_id_1))
        input_mask_1 = [1]*len(source_id_1) + [0]*(self.max_source_len-len(source_id_1))
        input_id_1 = self.__pad(source_id_1, self.max_source_len)

        source_id_2 = self.__trunk([self.cls_id] + feature["text2_id"], self.max_source_len)
        segment_id_2 = [0]*len(source_id_2) + [0]*(self.max_source_len-len(source_id_2))
        input_mask_2 = [1]*len(source_id_2) + [0]*(self.max_source_len-len(source_id_2))
        input_id_2 = self.__pad(source_id_2, self.max_source_len)

        pos1 = feature['pos1']
        pos2 = feature['pos2']
        item_num = feature['item_num']
        return input_id_1, input_mask_1, segment_id_1, \
                    input_id_2, input_mask_2, segment_id_2, pos1, pos2, item_num


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for i, x in enumerate(zip(*batch)):
        if i <= 5:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
        else:
            batch_tensors.append(np.asarray(x))
    return batch_tensors


def read_data(filename):
    print("Reading data from {} ...".format(filename))
    with open(filename) as fin:
        data = json.load(fin)
    return data


def tokenize_data(entangled_data, tokenizer, filename):
    if os.path.exists(filename):
        print("Loading tokenized data ...")
        data = torch.load(filename)
    else:
        print("Tokenizing data ...")
        data = []
        for i, item in tqdm(enumerate(entangled_data)):
            new_item = []
            for j, utterance in enumerate(item):
                # speaker = utterance['speaker']
                msg = utterance['text']
                label = utterance['label']
                # tokenized_msg = tokenizer.tokenize(msg)
                tokenized_msg = msg.split()
                new_item.append({
                    'msg': tokenized_msg, 
                    'label': label,
                    'item_num': i, 
                    'position': j,
                })
            data.append(new_item)
        torch.save(data, filename)
    return data    


def flatten_utterance(data, tokenizer, filename):
    if os.path.exists(filename):
        new_data, labels = torch.load(filename)
    else:
        labels = {}
        for item in data:
            one_label = []
            item_num = -1
            for utterance in item:
                item_num = utterance['item_num']
                label = utterance['label']
                one_label.append(label)
            labels[item_num] = one_label
        
        new_data = []
        for item in tqdm(data):
            for i in range(1, len(item)):
                for j in range(i):
                    msg1 = item[j]
                    msg2 = item[i]
                    text1 = msg1['msg']
                    text2 = msg2['msg']
                    msg1_pos = msg1['position']
                    msg2_pos = msg2['position']
                    item_num = msg1['item_num']
                    new_data.append({
                        'text1_id': tokenizer.convert_tokens_to_ids(text1), 
                        'text2_id': tokenizer.convert_tokens_to_ids(text2), 
                        'pos1': msg1_pos, 
                        'pos2': msg2_pos, 
                        'item_num': item_num
                    })
        torch.save([new_data, labels], filename)
    return new_data, labels


def get_model_and_tokenizer(args):
    if args.model_path is not 'none':
        model = BertModel.from_pretrained(args.model_path)
    else:
        model = BertModel.from_pretrained(args.model_config)
    tokenizer = BertTokenizer.from_pretrained(args.model_config)
    return model, tokenizer
        

def evaluate(args, model, tokenizer, data, labels, res_filename, logger):
    if args.fp16:
        from apex import amp 
    else:
        amp = None
    
    if amp:
        model = amp.initialize(model, opt_level=args.fp16_opt_level)
    
    eval_dataset = DatasetForBertEval(
        features=data, max_source_len=args.max_source_len,
        max_len=args.max_len,
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
    all_item_nums = None
    all_probs = None
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = [item.to(args.device) if i <= 5 else item for i, item in enumerate(batch)]
            input_id_1, input_mask_1, segment_id_1, \
                        input_id_2, input_mask_2, segment_id_2, pos1, pos2, item_num = batch
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
                all_pos1 = pos1
                all_pos2 = pos2 
                all_item_nums = item_num
            else:
                all_probs = np.append(all_probs, probs.detach().cpu().numpy(), axis=0)
                all_pos1 = np.append(all_pos1, pos1, axis=0)
                all_pos2 = np.append(all_pos2, pos2, axis=0) 
                all_item_nums = np.append(all_item_nums, item_num, axis=0)
    
    predictions = parse_predictions(all_probs, all_pos1, all_pos2, all_item_nums)
    with open(res_filename, 'w') as fout:
        json.dump([predictions, labels], fout, indent=4)


def parse_predictions(probs, all_pos1, all_pos2, all_item_num):
    preds_dict = {}
    print("Parsing Predictions ... ")
    for i in tqdm(range(len(all_item_num))):
        prob = round(float(probs[i]), 4)
        pos1 = int(all_pos1[i])
        pos2 = int(all_pos2[i])
        item_num = int(all_item_num[i])
        if item_num not in preds_dict:
            preds_dict[item_num] = {}
        if pos1 not in preds_dict[item_num]:
            preds_dict[item_num][pos1] = {}
        if pos2 not in preds_dict[item_num]:
            preds_dict[item_num][pos2] = {}
        preds_dict[item_num][pos1][pos2] = prob
        preds_dict[item_num][pos2][pos1] = prob
    return preds_dict


def prepare(args):
    if args.model_path is not 'none':
        args.current_time = re.findall(".*(2021.+?)/.*", args.model_path)[0]
        checkpoint = re.findall(".*ckpt-(.+?)/.*", args.model_path)[0]
        args.result_dir = os.path.join(args.output_dir, args.data_name, \
                                            args.aug_type, args.current_time, args.result_dir, checkpoint)
    else:
        args.current_time = strftime("%Y-%b-%d-%H_%M_%S", gmtime())
        args.result_dir = os.path.join(args.output_dir, args.data_name, \
                                            args.aug_type, args.current_time, args.result_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    args.output_dir = os.path.join(args.output_dir, args.data_name, args.aug_type, args.current_time)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.cached_input_dir = os.path.join(args.cached_input_dir, args.data_name, args.aug_type)
    if not os.path.exists(args.cached_input_dir):
        os.makedirs(args.cached_input_dir)
    
    # define the logger
    logger_name = os.path.join(args.output_dir, "{}_log.txt".format(args.data_type))
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--data_type', type=str, choices=['test', 'dev'], default='dev')
    parser.add_argument('--data_name', type=str, choices=['movie', 'politics', 'iphones', 'gadgets'], default='movie')
    parser.add_argument('--aug_type', type=str, choices=['same_speaker', 'filtered_same_speaker', 'mix', 'gold', 'no_train'], default='same_speaker')
    parser.add_argument('--model_config', type=str, default='bert-base-uncased')
    parser.add_argument('--model_path', type=str, default='none')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--max_source_len', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=32)
    parser.add_argument('--cached_input_dir', type=str, default='cached_input')
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
    
    filename = os.path.join(os.path.join(args.data_dir, args.data_name), 'tokenized', '{}.json'.format(args.data_type))
    entangled_data = read_data(filename)

    cached_tokenized_filename = os.path.join(args.cached_input_dir, "tokenized_{}.pt".format(args.data_type))
    tokenized_data = tokenize_data(entangled_data, tokenizer, cached_tokenized_filename)
    cached_flat_data_filename = os.path.join(args.cached_input_dir, "flat_tokenized_{}.pt".format(args.data_type)) 
    data, labels = flatten_utterance(tokenized_data, tokenizer, cached_flat_data_filename)

    model.to(args.device)

    res_filename = os.path.join(args.result_dir, '{}_res.pk'.format(args.data_type))
    evaluate(args, model, tokenizer, data, labels, res_filename, logger)
    


if __name__ == "__main__":
    main()

