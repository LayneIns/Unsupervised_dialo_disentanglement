import torch 
import torch.nn as nn 
import torch.functional as F 
import numpy as np 


class MatchModel(nn.Module):
    def __init__(self, rnn_cell, token_num, word_emb, hidden_size, emb_size, max_context_len, bidirectional=True, max_session_number=4, inference=False):
        super(MatchModel, self).__init__()
        self.message_encoder = MessageEncoder(rnn_cell, token_num, word_emb, hidden_size, emb_size, bidirectional=bidirectional)
        self.max_context_len = max_context_len
        self.attention_liearn = nn.Linear(hidden_size, 1)
        self.max_session_number = max_session_number
        self.inference = inference

    def pad_item(self, context, length, hidden_size, device):
        zero_item = torch.zeros([length, hidden_size], dtype=torch.float32, device=device)
        padded_context = torch.cat([context, zero_item], dim=0)
        return padded_context
    
    def build_context_batch(self, context, context_len, max_context_len):
        start = 0
        end = 0
        context_batch = []
        for one_len in context_len:
            start = end
            end = end + one_len
            context_for_a_sample = context[start:end]
            device = torch.device("cuda:{}".format(context_for_a_sample.get_device()) if torch.cuda.is_available() else "cpu")

            length, hidden_size = context_for_a_sample.size()
            length = max_context_len - length

            context_with_padding = self.pad_item(context_for_a_sample, length, hidden_size, device)
            context_batch.append(context_with_padding.unsqueeze(0))
        context_batch = torch.cat(context_batch, dim=0)
        return context_batch
    
    def get_decision(self, context, message, decision_sequence, session_number):
        batch_size = context.size()[0]
        ret_value = []
        for i in range(batch_size):
            one_context = context[i]
            one_message = message[i]
            one_decision = decision_sequence[i]
            value = []
            for k in range(session_number[i]):
                step_decision = one_decision == k
                step_context = one_context[step_decision]

                weight = self.attention_liearn(step_context).transpose(1, 0)
                context_attention_score = nn.Softmax(dim=1)(weight)
                attended_context_vector = torch.mm(context_attention_score, step_context)
                score = torch.mm(attended_context_vector, one_message.view(-1, 1))
                value.append(score.squeeze())
            max_num = max(value)
            max_value_index = value.index(max_num)
            ret_value.append((max_num, max_value_index))
        return ret_value
    
    def get_decision_v2(self, context, message, decision_sequence, session_number):
        batch_size = context.size()[0]
        ret_value = []
        new_session_value = []
        for i in range(batch_size):
            one_context = context[i]
            one_message = message[i]
            one_decision = decision_sequence[i]
            value = []
            new_session_weight = self.attention_liearn(one_context).transpose(1, 0)
            context_attention_score = nn.Softmax(dim=1)(new_session_weight)
            attended_context_vector = torch.mm(context_attention_score, one_context)
            score = torch.mm(attended_context_vector, one_message.view(-1, 1))
            new_session_value.append(score.squeeze())
            for k in range(session_number[i]):
                step_decision = one_decision == k
                step_context = one_context[step_decision]

                weight = self.attention_liearn(step_context).transpose(1, 0)
                context_attention_score = nn.Softmax(dim=1)(weight)
                attended_context_vector = torch.mm(context_attention_score, step_context)
                score = torch.mm(attended_context_vector, one_message.view(-1, 1))
                value.append(score.squeeze())
            max_num = max(value)
            max_value_index = value.index(max_num)
            ret_value.append((max_num, max_value_index))
        return ret_value, new_session_value
    
    def get_decision_v3(self, context, message, decision_sequence, session_number, detect_new=False):
        if detect_new:
            batch_size = context.size()[0]
            new_session_value = []
            for i in range(batch_size):
                one_context = context[i]
                one_message = message[i]
                one_decision = decision_sequence[i]
                value = []
                new_session_weight = self.attention_liearn(one_context).transpose(1, 0)
                context_attention_score = nn.Softmax(dim=1)(new_session_weight)
                attended_context_vector = torch.mm(context_attention_score, one_context)
                score = torch.mm(attended_context_vector, one_message.view(-1, 1))
                new_session_value.append(score.squeeze())
            return new_session_value
        else:
            batch_size = context.size()[0]
            ret_value = []
            for i in range(batch_size):
                one_context = context[i]
                one_message = message[i]
                one_decision = decision_sequence[i]
                value = []
                for k in range(session_number[i]):
                    step_decision = one_decision == k
                    step_context = one_context[step_decision]
                    weight = self.attention_liearn(step_context).transpose(1, 0)
                    context_attention_score = nn.Softmax(dim=1)(weight)
                    attended_context_vector = torch.mm(context_attention_score, step_context)
                    score = torch.mm(attended_context_vector, one_message.view(-1, 1))
                    value.append(score.squeeze())
                max_num = max(value)
                max_value_index = value.index(max_num)
                ret_value.append((max_num, max_value_index))
            return ret_value


    
    def build_context_batch_with_attention(self, context, context_len):
        start = 0
        end = 0
        context_batch = []
        for one_len in context_len:
            start = end
            end = end + one_len
            context_for_a_sample = context[start:end]
            weight = self.attention_liearn(context_for_a_sample).transpose(1, 0)
            context_attention_score = nn.Softmax(dim=1)(weight)
            attended_context_vector = torch.mm(context_attention_score, context_for_a_sample)
            context_batch.append(attended_context_vector)
        context_batch = torch.cat(context_batch, dim=0)
        return context_batch

    def forward(self, context, context_seq_len, context_len, current_message=None, 
                    current_message_seq_len=None, max_context_len=-1):
        # print(context.size())
        # print(context_seq_len.size())
        # print(context_len.size())
        # print(current_message.size())
        # print(current_message_seq_len.size())
        # print("Sum: {}".format(sum(context_len)))
        # print("=================1")
        # input()
        context_vector = self.message_encoder(context, context_seq_len)
        if self.inference:
            # context_batch = self.build_context_batch(context_vector, context_len, max_context_len)
            return context_vector
        else:
            context_batch = self.build_context_batch_with_attention(context_vector, context_len)
            current_message_vector = self.message_encoder(current_message, current_message_seq_len)

            batch_size, hidden_size = context_batch.size()
            logit = torch.bmm(
                context_batch.view(batch_size, 1, hidden_size), current_message_vector.view(batch_size, hidden_size, 1)
            ).squeeze(-1)
            pos_output = nn.Sigmoid()(logit)
            neg_output = 1 - pos_output
            output = torch.cat([neg_output, pos_output], dim=1)
            return output
        


class MessageEncoder(nn.Module):
    def __init__(self, rnn_cell, token_num, word_emb, hidden_size, emb_size, bidirectional, dropout_rate=0.1):
        super(MessageEncoder, self).__init__()
        self.word_emb = word_emb
        self.word_emb_matrix = nn.Embedding(token_num, emb_size)
        self.init_embedding()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.rnn_cell = rnn_cell
        self.bidirectional = bidirectional

        if self.rnn_cell == 'lstm':
            self.encoder = nn.LSTM(emb_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional)
        elif self.rnn_cell == 'gru':
            self.encoder = nn.GRU(emb_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional)
        else:
            raise NotImplementedError("Message encoder {} not implemented".format(self.rnn_cell))
        
        if self.bidirectional:
            self.output_linear = nn.Linear(hidden_size*2, hidden_size)
    
    def init_embedding(self):
        if self.word_emb is None:
            self.word_emb_matrix.weight.data.uniform_(-0.1, 0.1)
        else:
            self.word_emb_matrix.weight.data.copy_(torch.from_numpy(self.word_emb))
        
    def forward(self, input_data, seq_len):
        embeded_input = self.word_emb_matrix(input_data)
        embeded_input = self.dropout(embeded_input)
        
        packed_ = torch.nn.utils.rnn.pack_padded_sequence(embeded_input, seq_len.cpu(), batch_first=True, enforce_sorted=False)

        if self.rnn_cell == 'lstm':
            _, hidden_state = self.encoder(packed_)
            h_state = hidden_state[0]
        elif self.rnn_cell == 'gru':
            _, h_state = self.encoder(packed_)

        if self.bidirectional:
            h_state_cat = torch.cat([h_state[0], h_state[1]], dim=1)
            output_vec = self.output_linear(h_state_cat)
        return output_vec