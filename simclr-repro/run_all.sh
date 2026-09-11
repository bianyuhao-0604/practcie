#!/usr/bin/env bash
set -e

# ===== 单卡轻量复现 =====
python train.py \
  --dataset cifar10 --arch resnet18 \
  --batch_size 512 --epochs 200 \
  --temperature 0.5 --color_s 0.5 \
  --warmup_epochs 10 --amp

python linear_eval.py --ckpt runs/simclr_cifar10_last.pth

# ===== 多卡 DDP（4×GPU, 有效 batch=2048）=====
# torchrun --standalone --nproc_per_node=4 train.py \
#   --batch_size 2048 --epochs 200 --temperature 0.5 --amp
