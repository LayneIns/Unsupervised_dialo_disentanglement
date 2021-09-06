import sys 
import os 
from tqdm import tqdm 

from data_loader import DatasetforDisentanglement
import utils
import policy

import torch
import torch.nn as nn
import torch.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader, SequentialSampler
from transformers import get_linear_schedule_with_warmup


def supervised_trainer(args, model, training_data, coherency_reward, speaker_reward, logger, optimizer=None):
    model.to(args.device)

    model.load_state_dict(torch.load(args.init_checkpoint))
    # if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)
        
    if args.n_gpu == 0 or args.no_cuda:
        training_batch_size = args.per_gpu_batch_size
    else:
        training_batch_size = args.per_gpu_batch_size * args.n_gpu
    
    num_training_steps = int(args.epochs * len(training_data) / training_batch_size)

    train_dataset = DatasetforDisentanglement(
        training_data, coherency_reward, speaker_reward, args.seq_max_len, 
            num_training_steps*training_batch_size, args.pad_id)
    
     # Train!
    logger.info("  ***** Running training *****  *")
    logger.info("  Num examples = %d", len(training_data))
    logger.info("  Num Epochs = %.2f", len(train_dataset) / len(training_data))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", training_batch_size)
    logger.info("  Total optimization steps = %d", num_training_steps)

    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=training_batch_size,
        collate_fn=utils.batch_list_to_batch_tensors)

    train_iterator = tqdm(
        train_dataloader, initial=0,
            desc="Iter (loss=X.XXX, lr=X.XXXXXXX)")
    
    model.train()
    model.zero_grad()

    global_step = 0
    logging_loss = 0.
    # loss_func = nn.CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps, last_epoch=-1)

    for batch in train_iterator:
        context_tensor, context_seq_len_tensor, context_len, coherency_reward, speaker_reward = batch
        context_tensor = context_tensor.to(args.device)
        # context_seq_len_tensor = context_seq_len_tensor.to(args.device)
        context_len = context_len.to(args.device)
        max_context_len = max(context_len)
        context_vector = model.module.message_encoder(context_tensor, context_seq_len_tensor)
        context_batch = model.module.build_context_batch(context_vector, context_len, max_context_len)
        
        batch_size = context_len.size()[0]

        decision_sequence = torch.zeros([batch_size, 1], dtype=torch.int64, device=args.device)
        session_number = torch.ones([batch_size, 1], dtype=torch.int64, device=args.device)
        action_list = [[[None, None, None]] for _ in range(batch_size)]
        for step in range(1, max_context_len):
            current_context = context_batch[:, 0:step, :]
            current_message = context_batch[:, step, :]
            ret_value = model.module.get_random_decision(current_context, current_message, decision_sequence, session_number)
            action_list = policy.update_action_list(action_list, ret_value)
            current_decision = []
            for i in range(batch_size):
                action_type, action, prob, new_session_prob = ret_value[i]
                if action_type == 'new':
                    if session_number[i] < args.max_session_number:
                        current_decision.append(session_number[i].data.item())
                        session_number[i] += 1
                    else:
                        current_decision.append(action)
                elif action_type == 'select':
                    current_decision.append(action)
            current_decision = torch.tensor(current_decision, dtype=torch.int64, device=args.device).view(batch_size, 1)
            decision_sequence = torch.cat([decision_sequence, current_decision], dim=1)

        policy_loss = policy.get_reward_loss(args.coherency_reward_weight, action_list, decision_sequence, coherency_reward, speaker_reward, context_len)
        policy_loss = sum(policy_loss)/len(policy_loss)

        logging_loss += policy_loss

        train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (policy_loss.item(), scheduler.get_last_lr()[0]))
        
        policy_loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1

        if global_step % args.logging_steps == 0:
            logger.info(" Step [%d ~ %d]: %.2f", global_step - args.logging_steps, global_step, logging_loss)
            logging_loss = 0.0
        if args.save_steps > 0 and \
                (global_step % args.save_steps == 0 or global_step == num_training_steps):

            save_path = os.path.join(args.cache_dir, "ckpt-%d.pkl" % global_step)
            torch.save(model.state_dict(), save_path)
            logger.info("Saving model checkpoint %d into %s", global_step, save_path)