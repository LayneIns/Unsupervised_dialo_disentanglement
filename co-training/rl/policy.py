import os 
import sys 
import numpy as np 
import torch 


def update_action_list(action_list, current_actions):
    for i in range(len(action_list)):
        action_list[i] += [current_actions[i]]
    return action_list


def get_coherency_reward(coherency_reward, reward_type='avg', message_type='new'):
    if reward_type == 'avg':
        if message_type == 'new':
            return 0.5 - np.average(coherency_reward)
        elif message_type == 'select':
            return np.average(coherency_reward) - 0.5
    elif reward_type == 'max':
        if message_type == 'new':
            return 0.5 - np.amax(coherency_reward)
        elif message_type == 'select':
            return np.amax(coherency_reward) - 0.5


def get_speaker_reward(speaker_reward, message_type='new', scaling_lambda=-1):
    if message_type == 'new':
        if np.sum(speaker_reward) > 0:
            return -1
        else:
            return 0
    elif message_type == 'select':
        if np.sum(speaker_reward) > 0:
            return 1
        else:
            return scaling_lambda


def get_reward_loss(coherency_reward_weight, action_list, decision_sequence, coherency_reward, speaker_reward, context_len):
    batch_size = len(action_list)
    total_loss = []
    for i in range(batch_size):
        one_loss = 0.
        for j in range(1, context_len[i]):
            action_type, action, prob, new_session_probs = action_list[i][j]
            decision = decision_sequence[i][j].data.item()
            if action_type == 'new':
                step_coherency_matrix = coherency_reward[i][j, 0:j]
                new_action_coherency_reward = get_coherency_reward(step_coherency_matrix, reward_type='avg', message_type='new')
                step_speaker_matrix = speaker_reward[i][j, 0:j]
                new_action_speaker_reward = get_speaker_reward(step_speaker_matrix, message_type='new')
                step_reward = coherency_reward_weight * new_action_coherency_reward + (1-coherency_reward_weight) * new_action_speaker_reward
                step_loss = step_reward * (-torch.log(prob[0][0]))
                one_loss += step_loss
            elif action_type == 'select':
                step_choice = decision_sequence[i] == action
                step_choice = step_choice.cpu().numpy()
                step_choice = step_choice[0:j]
                step_coherency_matrix = coherency_reward[i][j][0:j][step_choice]
                step_speaker_matrix = speaker_reward[i][j][0:j][step_choice]
                new_action_coherency_reward = get_coherency_reward(step_coherency_matrix, reward_type='avg', message_type='select')
                new_action_speaker_reward = get_speaker_reward(step_speaker_matrix, message_type='select')
                step_reward = coherency_reward_weight * new_action_coherency_reward + (1-coherency_reward_weight) * new_action_speaker_reward
                step_loss = step_reward * (-torch.log(prob[0][1] * new_session_probs[0][1]))
                one_loss += step_loss
        total_loss.append(one_loss)

    return total_loss



