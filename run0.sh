#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py  \
    --name __unet__ \
    --model unet \
    --datadir /srv/data/ckames/homodynenet/mat/full \
    --epoch_length 500 \
    --batch_size 32 \
    --patch 128 \
    --suffix _local \
    --layers 1 2 3 5 8 \
    --lr 5e-5 \
    --min_lr 3e-6 \
    --patience 16 \
    --weight_decay 1e-5 \
    --gamma 0.5
    
