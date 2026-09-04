#!/bin/bash
#
#SBATCH --ntasks=3
#SBATCH --partition=shared-cpu
#SBATCH --time=2-0
#SBATCH --error=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/err/%j.err
#SBATCH --output=/nfs/home/students/t.reim/bachelor/pytorchtest/slurm/out/%j.out
#SBATCH --job-name=contact_maps
#SBATCH --mem=40G

srun python /nfs/home/students/t.reim/bachelor/pytorchtest/data/get_contact.py
