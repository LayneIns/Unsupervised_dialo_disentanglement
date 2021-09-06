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


def tokenize_data(data, tokenizer):
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
    
    print("{} tokenized data read.".format(len(tokenized_data)))
    return tokenized_data


def convert_data_to_id(data, word_dict):
    print("Converting data to id ...")
    data_id = []
    for item in tqdm(data):
        new_item = []
        for message in item:
            text = message['message']
            label = message['label']
            speaker = message['speaker']
            message_id = [word_dict.get(token, word_dict.get('<UNK>')) for token in text]
            new_item.append({
                'message': message_id,
                'label': label,
                'speaker': speaker
            })
        data_id.append(new_item)

    print("{} data id read.".format(len(data_id)))
    return data_id


def get_max_context_length(data_id):
    print("Retrieving the maximum context length ...")
    context_length = 0
    for item in tqdm(data_id):
        context_length = max(len(item), context_length)
    print("Max context length: {}".format(context_length))
    return context_length


def parse_reward(reward, data):
    print("Parsing reward item ...")
    coherency_reward = []
    speaker_reward = []
    for i in tqdm(range(len(data))):
        data_item = data[i]
        coherency_matrix = reward[i]['reward']
        assert coherency_matrix.shape[0] == len(data_item)
        speaker_matrix = np.zeros_like(coherency_matrix, dtype=float)
        for i in range(len(data_item)):
            for j in range(len(data_item)):
                speaker_1 = data_item[i]['speaker']
                speaker_2 = data_item[j]['speaker']
                val = 1 if speaker_1 == speaker_2 else 0
                speaker_matrix[i][j] = val
        coherency_reward.append(coherency_matrix)
        speaker_reward.append(speaker_matrix)
    return coherency_reward, speaker_reward


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], np.ndarray):
            batch_tensors.append(np.array(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def batch_list_to_batch_tensors(batch):
    new_batch = list(zip(*batch))
    context, context_seq_len, _, context_len, coherency_reward, speaker_reward = new_batch

    context = list(itertools.chain(*context))
    context_tensor = torch.tensor(context, dtype=torch.long)
    context_seq_len = list(itertools.chain(*context_seq_len))
    context_seq_len_tensor = torch.tensor(context_seq_len, dtype=torch.long)

    context_len = torch.tensor(context_len, dtype=torch.long)

    return context_tensor, context_seq_len_tensor, context_len, coherency_reward, speaker_reward
