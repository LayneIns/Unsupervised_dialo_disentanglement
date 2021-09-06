import sys 
import os 
from tqdm import tqdm 

from data_loader import DatasetforMM
import utils

import torch
import torch.nn as nn
import torch.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader, SequentialSampler


def supervised_trainer(args, model, training_data, logger, optimizer=None):

    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.n_gpu == 0 or args.no_cuda:
        training_batch_size = args.per_gpu_batch_size
    else:
        training_batch_size = args.per_gpu_batch_size * args.n_gpu
    num_training_steps = int(args.epochs * len(training_data) / training_batch_size)
    
    train_dataset = DatasetforMM(
        training_data, args.seq_max_len, num_training_steps*training_batch_size, args.pad_id)

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
    loss_func = nn.CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    for batch in train_iterator:
        batch_tensors = tuple(t.to(args.device) for t in batch)
        context, context_seq_len, context_len, current_message, current_message_seq_len, label = batch_tensors
        outputs = model(context, context_seq_len, context_len, current_message, current_message_seq_len)
        loss = loss_func(outputs, label)
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        
        train_iterator.set_description('Iter (loss=%5.3f)' % loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logging_loss += loss.item()
        global_step += 1

        if args.logging_steps > 0 and global_step % args.logging_steps == 0:
            logger.info(" Step [%d ~ %d]: %.2f", global_step - args.logging_steps, global_step, logging_loss)
            logging_loss = 0.0

        if args.save_steps > 0 and \
                (global_step % args.save_steps == 0 or global_step == num_training_steps):

            save_path = os.path.join(args.cache_dir, "ckpt-%d.pkl" % global_step)
            torch.save(model.state_dict(), save_path)
            logger.info("Saving model checkpoint %d into %s", global_step, save_path)
