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
#SBATCH --job-name=ppi-2dbaseline
#SBATCH --mem-per-gpu=40G

srun python /nfs/home/students/t.reim/bachelor/pytorchtest/main.py \
-data gold_stand \
-model baseline2d \
-lr 0.001 \
-max 1000 \
-epoch 25 \
-batch 40 \
-sub \
-subsize 0.5 \
-emb \
-emb_dim 5120 \
-wandb \
-run baseline2d_test_t48 