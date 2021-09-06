import os 
import sys 
import numpy as np 
import torch 


class DatasetforDisentanglement(torch.utils.data.Dataset):
    def __init__(self, training_data, coherency_reward, speaker_reward, seq_max_len, num_training_instances, pad_id):
        self.training_data = training_data
        self.coherency_reward = coherency_reward
        self.speaker_reward = speaker_reward
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
        speaker_reward = self.speaker_reward[idx]
        coherency_reward = self.coherency_reward[idx]
        messages = []
        labels = []
        message_seq_len = []
        for item in example:
            text = item['message']
            label = item['label']
            text = self.__trunk(text, self.seq_max_len)
            text_seq_len = len(text)
            text = self.__pad(text, self.seq_max_len)
            messages.append(text)
            message_seq_len.append(text_seq_len)
            labels.append(label)
        return messages, message_seq_len, labels, len(messages), coherency_reward, speaker_reward