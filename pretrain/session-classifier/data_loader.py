import os 
import sys 
import numpy as np 
import torch 


class DatasetforMM(torch.utils.data.Dataset):
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
        context_history = example['messages']
        current_message = example['current_message'][1]
        label_id = example['label']

        current_message = self.__trunk(current_message, self.seq_max_len)
        current_message_seq_len = len(current_message)
        current_message = self.__pad(current_message, self.seq_max_len)
        
        context = [item[1] for item in context_history]
        context = [self.__trunk(item, self.seq_max_len) for item in context]
        context_seq_len = [len(x) for x in context]
        context = [self.__pad(item, self.seq_max_len) for item in context]
        context_len = len(context)

        return context, context_seq_len, context_len, current_message, current_message_seq_len, label_id