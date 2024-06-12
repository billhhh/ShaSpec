#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=Eval_mode1_BraTS18_ShaSpec_[80,160,160]_SGD_b1_lr-2_alpha.1_beta.02_trainOnly_rand_mode

CUDA_VISIBLE_DEVICES=$1 python eval.py \
--input_size=80,160,160 \
--num_classes=3 \
--data_list=BraTS18/BraTS18_test.csv \
--weight_std=True \
--restore_from=snapshots/BraTS18_ShaSpec_[80,160,160]_SGD_b1_lr-2_alpha.1_beta.02_trainOnly_rand_mode/final.pth \
--mode=0,1,2,3 > logs/${time}_train_${name}.log
