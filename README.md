
# Unsupervised-dialo-disentanglement

This respository contains the source code of the paper [Unsupervised Conversation Disentanglement through Co-Training]() which is published in EMNLP 2021. In this work, we explore training a conversation disentanglement model without referencing any human annotations. Our method is built upon the deep co-training algorithm. Experimental results on the large Movie Dialogue Dataset demonstrate that our proposed approach achieves competitive performance compared to previous supervised methods. The Movie Dialogue Dataset can be downloaded from [here](https://github.com/LayneIns/E2E-dialo-disentanglement).

![Illustration of our proposed co-training framework](./cotraining-framework.png)
<center>*Illustration of our proposed co-training framework*</center>

## Pre-training

### Message-pair Classifier
To initialize the Message-pair Classifier, we can build the pseudo dataset by retrieving the message pairs from the same speaker in a conversation and by using the query-output pairs predicted by DialoGPT. Generally the format of the pseudo dataset is as below:

	[
		["You're a big kid now. ", "It's still sick. ", 0], 
		["We didnt do nothin special I can  remember.  Just talked, is all. ", "Is that the truth? ", 0],
		["What's that? ", "It's beautiful. Thanks. ", 1],
		...
	 ]

You can finetune the message-pair classifier with the built pseudo dataset:
```
cd ./pretrain/message-pair-classifier/
python bert_train.py --data_name movie --aug_type same_speaker --model_config bert-base-uncased --num_training_epochs 2 --logging_steps 100 --save_steps 800 --fp16
```
You can evaluate the performance of the finetuned model by:
```
cd ./pretrain/message-pair-classifier/
python bert_eval_disentangle.py --data_type dev --data_name movie --aug_type same_speaker --model_path output/movie/same_speaker/ckpt-800/ --per_gpu_eval_batch_size 128 --fp16
```

### Session Classifier
Similarly as the message-pair classifier, we can build the pseudo data to initialize the Session Classifier. The format of the pseudo dataset is as:

	[
		{
			"messages": [[speaker_1, text_1], [speaker_2, text_2], ...],
			"current_message": [speaker_n, text_n],
			"label": 1
		},
		{
			"messages": [[speaker_1, text_1], [speaker_2, text_2], ...],
			"current_message": [speaker_n, text_n],
			"label": 0
		},
		...
	 ]

You can pre-train the Session Classifier with the pseudo dataset as:
```
cd ./pretrain/session-classifier/
python main.py --bidirectional --per_gpu_batch_size 1024 --save_steps 2500 --logging_steps 250 --epochs 20 --lr 1e-3
```
You can evaluate the performance as:
```
cd ./pretrain/session-classifier/
python disentangle_inference.py --output_dir output/movie/ --checkpoint -1 --bidirectional --per_gpu_batch_size 512
```
## Co-training

Before co-training, you have to obtain the local relation between every message pairs in a conversation in the training set. You can obtain it by:
```
cd ./co-training/bert-reward/
python bert-eval.py --model_path checkpoint/1/ckpt-16000/ --fp16 --read_cached_input --per_gpu_eval_batch_size 4096 --round 1
```
This step will yield the reward files ``reward.pt`` which will be used in the RL training stage.

During the RL training stage, you can run:
```
cd ./co-training/rl/
python main.py --init_checkpoint initialize/1/ckpt-7500.pkl --word_dict initialize/word_dict.pt --bidirectional --per_gpu_batch_size 128 --lr 5e-6 --epochs 3 --coherency_reward_weight 0.6 --round 1
```
The ``--init_checkpoint`` argument specifies the saved checkpoint of the Session Classifier during the pre-training step. The ``--word_dict`` argument specifies the word dictionary used by the Session Classifier during the pre-training step.

You can iteratively run the pre-training step and the co-training step, where you can update the pseudo training dataset of the Message-pair Classifier with the  disentanglement results predicted during co-training step. And then the updated Message-pair Classifier will predict new local relations of message pairs.

You can test the performance by:
```
cd ./co-training/rl/
python disentangle_inference.py --output_dir rl-train/output/movie/1/ --checkpoint 100 --bidirectional --per_gpu_batch_size 512
```

## Citation

Coming soon!

## Q&A
If you have any questions about the paper and the code, please feel free to leave an issue or send me an email.
