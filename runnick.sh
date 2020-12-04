#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py  \
    --model unet3d \
    --datadir /srv/data/coopar6/microbleeds_ai/dataset_5 \
    --epoch_length 80\
    --epochs 1000\
    --batch_size 3 \
    --patch 64 256 256 \
    --layers 1 1 1 1 1 \
    --nrm batch \
    --lr 1e-4 \
    --early_stopping 0 \
    --act relu \
    --loss diceloss \
    --skip 0 \
    --nk 16 \
    --cin 2 \
    
