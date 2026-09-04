#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --partition=shared-cpu
#SBATCH --exclude=gpu01.exbio.wzw.tum.de
#SBATCH --time=2-0
#SBATCH --error=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/err/%j.err
#SBATCH --output=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/out/rfc_48_mean.out
#SBATCH --job-name=rfc_baseline
#SBATCH --mem=40G

srun python /nfs/home/students/t.reim/bachelor/pytorchtest/models/baselineRFC.py --model esm2_t48_15B --mean --layer 48 --components 20 --data_name gold_stand