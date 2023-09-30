#!/bin/bash

python main_select.py --net_name fedconv_invertup \
--dataset inat --split_type real \
--data_path path-to-dataset \
--num_local_clients 25 \
--max_communication_rounds 2000 \
--gpu_ids 0 \
--optimizer_type sgd --weight_decay 0 \
--lr 0.03 \
--drop_path 0 \
--agc 0.01 \
--batch_size 64 \
--save_model_avg \
--Pretrained --pretrained_dir path-to-pretrained-model