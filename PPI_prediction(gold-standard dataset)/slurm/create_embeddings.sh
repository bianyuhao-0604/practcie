#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=44GB
#SBATCH --cpus-per-task=10
#SBATCH --partition=shared-gpu
#SBATCH --nodelist=gpu02.exbio.wzw.tum.de
#SBATCH --exclude=gpu01.exbio.wzw.tum.de
#SBATCH --time=2-0
#SBATCH --error=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/err/%j.err
#SBATCH --output=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/out/%j.out
#SBATCH --job-name=create_embeddings

srun python /nfs/home/students/t.reim/bachelor/pytorchtest/data/extract.py esm2_t48_15B_UR50D /nfs/home/students/t.reim/bachelor/pytorchtest/data/swissprot/human_swissprot_oneliner_10k.fasta /nfs/scratch/t.reim/embeddings/esm2_t48_15B/mean --include mean
