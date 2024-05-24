#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=BraTS18_ShaSpec_[80,160,160]_SGD_b1_lr-2_alpha.1_beta.02_trainOnly

CUDA_VISIBLE_DEVICES=$1 python train_SS.py \
--snapshot_dir=snapshots/$name/ \
--input_size=80,160,160 \
--batch_size=1 \
--num_gpus=1 \
--num_steps=80000 \
--val_pred_every=2000 \
--learning_rate=1e-2 \
--num_classes=3 \
--num_workers=4 \
--train_list=BraTS18/BraTS18_train_all.csv \
--val_list=BraTS18/BraTS18_val.csv \
--random_mirror=True \
--random_scale=True \
--weight_std=True \
--train_only \
--reload_path=snapshots/tmp/final.pth \
--reload_from_checkpoint=False > logs/${time}_train_${name}.log


time=$(date "+%Y%m%d-%H%M%S")
name=BraTS18_ShaSpec_[80,160,160]_SGD_b1_lr-2_alpha.1_beta.02_trainOnly_rand_mode

CUDA_VISIBLE_DEVICES=$1 python train_SS.py \
--snapshot_dir=snapshots/$name/ \
--input_size=80,160,160 \
--batch_size=1 \
--num_gpus=1 \
--num_steps=180000 \
--val_pred_every=2000 \
--learning_rate=1e-2 \
--num_classes=3 \
--num_workers=4 \
--train_list=BraTS18/BraTS18_train_all.csv \
--val_list=BraTS18/BraTS18_val.csv \
--random_mirror=True \
--random_scale=True \
--weight_std=True \
--train_only \
--reload_path=snapshots/BraTS18_ShaSpec_[80,160,160]_SGD_b1_lr-2_alpha.1_beta.02_trainOnly/final.pth \
--reload_from_checkpoint=True \
--mode=random > logs/${time}_train_${name}.log
