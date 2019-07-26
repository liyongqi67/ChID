#!/bin/bash
set -ex

nohup python3 -u run_chid.py \
	--vocab_file ./chinese_wwm_pytorch/vocab.txt \
	--bert_config_file ./chinese_wwm_pytorch/bert_config.json \
	--init_checkpoint ./chinese_wwm_pytorch/pytorch_model.bin \
	--do_train \
	--do_predict \
	--train_file /home/share/liyongqi/ChID/raw_data/train.txt \
	--train_ans_file /home/share/liyongqi/ChID/raw_data/train_answer.csv \
	--predict_file /home/share/liyongqi/ChID/raw_data/dev.txt \
	--train_batch_size 16 \
	--predict_batch_size 16 \
	--learning_rate 2e-5 \
	--num_train_epochs 10.0 \
	--max_seq_length 256 \
	--output_dir ./output_model