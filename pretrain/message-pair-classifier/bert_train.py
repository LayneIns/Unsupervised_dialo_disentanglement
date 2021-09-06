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
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

import transformers 
from transformers import BertForNextSentencePrediction, BertTokenizer, BertModel
import apex 


class DatasetForBert(torch.utils.data.Dataset):
    def __init__(
            self, features, max_source_len, max_seq_len,
            cls_id, sep_id, pad_id, mask_id,
            offset, num_training_instances):
        self.features = features
        self.max_source_len = max_source_len
        self.max_seq_len = max_seq_len
        self.offset = offset
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.num_training_instances = num_training_instances
    
    def __len__(self):
        return int(self.num_training_instances)

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
        idx = (self.offset + idx) % len(self.features)
        feature = self.features[idx]

        source_id_1 = self.__trunk([self.cls_id] + feature["sent1"], self.max_source_len)
        segment_id_1 = [0]*len(source_id_1) + [0]*(self.max_source_len-len(source_id_1))
        input_mask_1 = [1]*len(source_id_1) + [0]*(self.max_source_len-len(source_id_1))
        input_id_1 = self.__pad(source_id_1, self.max_source_len)

        source_id_2 = self.__trunk([self.cls_id] + feature["sent2"], self.max_source_len)
        segment_id_2 = [0]*len(source_id_2) + [0]*(self.max_source_len-len(source_id_2))
        input_mask_2 = [1]*len(source_id_2) + [0]*(self.max_source_len-len(source_id_2))
        input_id_2 = self.__pad(source_id_2, self.max_source_len)

        label_id = feature["label"]
        return input_id_1, input_mask_1, segment_id_1, \
                    input_id_2, input_mask_2, segment_id_2, label_id


def read_data(filename):
    print("Reading data from {} ...".format(filename))
    with open(filename) as fin:
        data = json.load(fin)
    return data


def tokenize_data(train_data, tokenizer, filename):
    if os.path.exists(filename):
        print("Loading tokenized data ...")
        data = torch.load(filename)
    else:
        print("Tokenizing data ...")
        data = []
        for item in tqdm(train_data):
            sent1, sent2, label = item
            sent1_id = tokenizer.convert_tokens_to_ids(sent1.split()) 
            sent2_id = tokenizer.convert_tokens_to_ids(sent2.split())
            data.append({
                'sent1': sent1_id, 
                'sent2': sent2_id, 
                'label': label,
            })
        torch.save(data, filename)
    return data


def get_model_and_tokenizer(model_config):
    model = BertModel.from_pretrained(model_config)
    tokenizer = BertTokenizer.from_pretrained(model_config)
    return model, tokenizer


def prepare_for_training(args, model, checkpoint_state_dict, amp=None):
    # define the optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        if checkpoint_state_dict:
            amp.load_state_dict(checkpoint_state_dict['amp'])

    if checkpoint_state_dict:
        optimizer.load_state_dict(checkpoint_state_dict['optimizer'])
        model.load_state_dict(checkpoint_state_dict['model'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, optimizer


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for i, x in enumerate(zip(*batch)):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        if i == 6:
            batch_tensors.append(torch.tensor(x, dtype=torch.float))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def train(args, model, tokenizer, training_data, logger):
    if args.fp16:
        from apex import amp 
    else:
        amp = None
    
    checkpoint_state_dict = None

    model.to(args.device)
    model, optimizer = prepare_for_training(args, model, checkpoint_state_dict, amp=amp)
    
    # define the total batch size
    if args.n_gpu == 0 or args.no_cuda:
        per_node_train_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps
    else:
        per_node_train_batch_size = args.per_gpu_train_batch_size * args.n_gpu * args.gradient_accumulation_steps
        
    args.train_batch_size = per_node_train_batch_size
    global_step = 0

    # the total training steps
    if args.num_training_steps == -1:
        args.num_training_steps = int(args.num_training_epochs * len(training_data) / args.train_batch_size)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps, last_epoch=-1)

    if checkpoint_state_dict:
        scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler"])

    # dataset
    train_dataset = DatasetForBert(
        features=training_data,
        max_source_len=args.max_source_len, 
        max_seq_len=args.max_len, 
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, 
        pad_id=tokenizer.pad_token_id, mask_id=tokenizer.mask_token_id, 
        offset=args.train_batch_size * global_step, 
        num_training_instances=args.train_batch_size * args.num_training_steps,
    )

    # Train!
    logger.info("  ***** Running training *****  *")
    logger.info("  Num examples = %d", len(training_data))
    logger.info("  Num Epochs = %.2f", len(train_dataset) / len(training_data))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Batch size per node = %d", per_node_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.num_training_steps)
    
    # The training features are shuffled
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=per_node_train_batch_size // args.gradient_accumulation_steps,
        collate_fn=batch_list_to_batch_tensors)

    train_iterator = tqdm(
        train_dataloader, initial=global_step,
        desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=False)

    model.train()
    model.zero_grad()

    logging_loss = 0.0

    for step, batch in enumerate(train_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        labels = batch[6]
        inputs_1 = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    }
        if args.model_config.startswith('bert'):
            inputs_1['token_type_ids'] = batch[2]
            
        sent1_vec = model(**inputs_1)[1]

        inputs_2 = {'input_ids': batch[3],
                    'attention_mask': batch[4],
                    }
        if args.model_config.startswith('bert'):
            inputs_2['token_type_ids'] = batch[5]
            
        sent2_vec = model(**inputs_2)[1]

        score = torch.matmul(sent1_vec.unsqueeze(1), sent2_vec.unsqueeze(1).permute(0, 2, 1)).squeeze(-1).squeeze(-1)
        logits = nn.Sigmoid()(score)

        loss_fct = nn.BCELoss()
        loss = loss_fct(logits, labels)

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

        train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_last_lr()[0]))

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        logging_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logger.info(" Step [%d ~ %d]: %.2f", global_step - args.logging_steps, global_step, logging_loss)
                logging_loss = 0.0

            if args.save_steps > 0 and (global_step % args.save_steps == 0 or global_step == args.num_training_steps):
                save_path = os.path.join(args.cache_dir, "ckpt-%d" % global_step)
                os.makedirs(save_path, exist_ok=True)
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(save_path)
                logger.info("Saving model checkpoint %d into %s", global_step, save_path) 


def prepare(args):
    args.current_time = strftime("%Y-%b-%d-%H_%M_%S", gmtime())
    args.output_dir = os.path.join(args.output_dir, args.data_name, args.aug_type, args.current_time)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.cache_dir = os.path.join(args.output_dir, args.cache_dir)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    args.cached_input_dir = os.path.join(args.cached_input_dir, args.data_name, args.aug_type)
    if not os.path.exists(args.cached_input_dir):
        os.makedirs(args.cached_input_dir)
    
    # define the logger
    logger_name = os.path.join(args.output_dir, "log.txt")
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logger_name, filemode='a')
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    logger_msg = '\n========================='
    logger.info(logger_msg)
    logger.info(json.dumps(args.__dict__, sort_keys=True, indent=4))
    args.device = device

    if args.fp16:
        try:
            apex.amp.register_half_function(torch, 'einsum')
            apex.amp.register_float_function(torch, 'sigmoid')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--data_name', type=str, choices=['movie'], default='movie')
    parser.add_argument('--aug_type', type=str, choices=['same_speaker', 'filtered_same_speaker', 'mix', 'gold'], default='same_speaker')
    parser.add_argument('--model_config', type=str, default='bert-base-uncased')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--max_source_len', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_training_steps', type=int, default=-1)
    parser.add_argument('--num_training_epochs', type=int, default=1)
    parser.add_argument('--num_warmup_steps', type=int, default=1000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=32)
    parser.add_argument('--logging_steps', type=int, default=1000)
    parser.add_argument('--save_steps', type=int, default=8000)
    parser.add_argument('--cached_input_dir', type=str, default='cached_input')
    parser.add_argument('--cache_dir', type=str, default='cache')
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

    model, tokenizer = get_model_and_tokenizer(args.model_config)
    
    filename = os.path.join(args.data_dir, args.data_name, args.aug_type, 'train.json')
    train_data = read_data(filename)

    cached_tokenized_filename = os.path.join(args.cached_input_dir, "tokenized_train.pt")
    tokenized_data = tokenize_data(train_data, tokenizer, cached_tokenized_filename)

    model.to(args.device)

    train(args, model, tokenizer, tokenized_data, logger)


if __name__ == "__main__":
    main()

