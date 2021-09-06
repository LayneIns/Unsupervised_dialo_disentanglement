import os 
import sys 
from tqdm import tqdm 
import random 
import json 
import numpy as np 
import torch 
import itertools


def read_raw_data(data_path):
    print("Reading original data ...")
    with open(data_path) as fin:
        data = json.load(fin)
    print("{} data instances read from the original dataset.".format(len(data)))
    return data


def get_max_context_length(data, logger):
    print("Retrieving the maximum context length ...")
    context_length = 0
    for item in tqdm(data):
        context_length = max(len(item['messages']), context_length)
    logger.info("Max context length: {}".format(context_length))
    return context_length


def tokenize_data(data, cached_features_file, tokenizer):
    # if os.path.exists(cached_features_file):
    #     print("Loading tokenized data from cached file {}".format(cached_features_file))
    #     tokenized_data = torch.load(cached_features_file)
    # else:
    print("Tokenizing data ...")
    tokenized_data = []
    for item in tqdm(data):
        context_history = item['messages']
        current_message = item['current_message'][1].split()
        new_history = []
        for message in context_history:
            text = message[1].split()
            new_history.append([message[0], text, message[2]])
        tokenized_data.append({
            'messages': new_history,
            'current_message': [item['current_message'][0], current_message, item['current_message'][2]],
            'label': item['label']
        })
    # print("Saving tokenized data into cached file {}".format(cached_features_file))
    # torch.save(tokenized_data, cached_features_file)
    
    return tokenized_data


def get_word_dict(data, cached_features_file, min_token_freq, logger):
    if os.path.exists(cached_features_file):
        print("Loading word dictionary from cached file {}".format(cached_features_file))
        word_dict = torch.load(cached_features_file)
    else:
        print("Building word dict ...")
        word_cnt = dict()
        for item in tqdm(data):
            context_history = item['messages']
            current_message = item['current_message']
            for message in context_history:
                text = message[1]
                for token in text:
                    if token not in word_cnt:
                        word_cnt[token] = 0
                    word_cnt[token] += 1
            for token in current_message[1]:
                if token not in word_cnt:
                    word_cnt[token] = 0
                word_cnt[token] += 1
        word_dict = {}
        word_dict['<PAD>'] = 0
        word_dict['<UNK>'] = 1
        for key in word_cnt:
            if word_cnt[key] >= min_token_freq:
                word_dict[key] = len(word_dict)
        print("{} tokens in total, {} tokens in word dictionary.".format(len(word_cnt), len(word_dict)))
        logger.info("{} tokens in total, {} tokens in word dictionary.".format(len(word_cnt), len(word_dict)))

        print("Saving word dictionary into cached file {}".format(cached_features_file))
        torch.save(word_dict, cached_features_file)

    logger.info("{} tokens in word dictionary.".format(len(word_dict)))
    return word_dict


def convert_data_to_id(data, word_dict, cached_features_file, logger):
    if os.path.exists(cached_features_file):
        print("Loading training data id from cached file {}".format(cached_features_file))
        data_id = torch.load(cached_features_file)
    else:
        print("Converting data to id ...")
        data_id = []
        for item in tqdm(data):
            context_history = item['messages']
            current_message = item['current_message']

            current_message_id = []
            for token in current_message[1]:
                current_message_id.append(word_dict.get(token, word_dict.get('<UNK>')))

            context_history_id = []
            for message in context_history:
                speaker, text, one_label = message
                text_id = []
                for token in text: 
                    text_id.append(word_dict.get(token, word_dict.get('<UNK>')))
                context_history_id.append([speaker, text_id, one_label])
            data_id.append({
                'messages': context_history_id,
                'current_message': [item['current_message'][0], current_message_id, item['current_message'][2]],
                'label': item['label']
            })
        print("Saving data id into cached file {}".format(cached_features_file))
        torch.save(data_id, cached_features_file)

    logger.info("{} data id read.".format(len(data_id)))
    return data_id


def build_word_emb_matrix(word_dict, glove_filepath, emb_size, logger, cached_features_file):
    if os.path.exists(cached_features_file):
        print("Loading features from cached file {}".format(cached_features_file))
        word_emb_matrix = torch.load(cached_features_file)
    else:
        print("Building word embedding matrix ...")
        id_dict = {}
        for key, val in word_dict.items():
            id_dict[val] = key
        word_emb_matrix = np.random.normal(size=(len(word_dict), emb_size))
        cnt = 0
        with open(glove_filepath) as fin:
            for line in tqdm(fin):
                tokens = line.strip().split()
                token = " ".join(tokens[:-300])
                if token in word_dict:
                    emb = np.asarray([float(one_tok) for one_tok in tokens[-300:]])
                    word_emb_matrix[word_dict[token]] = emb
                    cnt += 1
        logger.info("{} out of {} tokens are initialized".format(cnt, len(word_dict)))
        torch.save(word_emb_matrix, cached_features_file)
    return word_emb_matrix


def batch_list_to_batch_tensors(batch):
    new_batch = list(zip(*batch))
    context, context_seq_len, context_len, current_message, current_message_seq_len, label_id = new_batch

    context = list(itertools.chain(*context))
    context_tensor = torch.tensor(context, dtype=torch.long)
    context_seq_len = list(itertools.chain(*context_seq_len))
    context_seq_len_tensor = torch.tensor(context_seq_len, dtype=torch.long)

    current_message_tensor = torch.tensor(current_message, dtype=torch.long)
    current_message_seq_len_tensor = torch.tensor(current_message_seq_len, dtype=torch.long)

    context_len = torch.tensor(context_len, dtype=torch.long)
    label_tensor = torch.tensor(label_id, dtype=torch.long)

    batch_tensors = [context_tensor, context_seq_len_tensor, context_len,
                        current_message_tensor, current_message_seq_len_tensor,
                        label_tensor]

    return batch_tensors
