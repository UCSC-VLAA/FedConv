#!/bin/bash

python main.py --net_name fedconv_invertup \
--dataset cifar10 --split_type split3 \
--gpu_ids 0 \
--optimizer_type sgd --weight_decay 0 \
--lr 0.03 \
--drop_path 0.1 \
--agc 0.01 \
--Pretrained --pretrained_dir /home/cihangxie/rx/FL/checkpoint/fed_conv3_silu_k9_nonnorm_agc.pth.tar \