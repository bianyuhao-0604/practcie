#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --partition=shared-gpu
#SBATCH --exclude=gpu01.exbio.wzw.tum.de
#SBATCH --time=2-0
#SBATCH --error=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/err/%j.err
#SBATCH --output=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/out/%j.out
#SBATCH --job-name=ppi-TUnA
#SBATCH --mem-per-gpu=40G

srun python /nfs/home/students/t.reim/bachelor/pytorchtest/main.py \
-data gold_stand \
-model TUnA \
-lr 0.001 \
-max 1000 \
-epoch 25 \
-batch 8 \
-sub \
-subsize 0.5 \
-es 6 \
-emb \
-emb_dim 5120 \
-wandb \
-heads 8 \
-dropout 0.2 \