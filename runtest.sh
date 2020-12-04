#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python eval.py  \
    --model unet3d \
    --datadir /srv/data/coopar6/microbleeds_ai/dataset_2 \
    --outdir /data/ubcitar/Nick/microbleeds_ai/mask_predictions/localfield/real_mb/ \
    --ckp /srv/data/coopar6/microbleeds_ai/log/paper/20201028_111731_playground/checkpoint_45000.pt \
    --nrm batch \