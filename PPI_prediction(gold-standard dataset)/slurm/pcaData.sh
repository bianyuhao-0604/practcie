#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --partition=shared-cpu
#SBATCH --exclude=gpu01.exbio.wzw.tum.de
#SBATCH --time=2-0
#SBATCH --error=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/err/%j.err
#SBATCH --output=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/out/%j.out
#SBATCH --job-name=rfc_baseline
#SBATCH --mem=40G

srun python /nfs/home/students/t.reim/bachelor/pytorchtest/data/pca_dataset.py --model esm2_t33_650 --mean --layer 33 --components 200