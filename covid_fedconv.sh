#!/bin/bash

python  main.py --net_name fedconv_base \
--dataset COVIDfl --split_type real_test \
--data_path /data1/rx/COVID-FL \
--gpu_ids 0 \
--Pretrained --pretrained_dir checkpoint/fed_final.pth.tar \
--lr 1.75e-4 \
--optimizer_type adamw \
--batch_size 64 \
--weight_decay 0.01 \
--mixup 0 --cutmix 0 \
--agc 0.01