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
#SBATCH --job-name=sweep
#SBATCH --mem-per-gpu=40G

srun python /nfs/home/students/t.reim/bachelor/pytorchtest/main.py -sweep