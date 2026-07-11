#!/bin/bash
#SBATCH --job-name=train_chem
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=8G

# The script takes arguments passed to sbatch
# Example: sbatch job_train.slurm --epochs 500 --batch_size 64

python3 train.py "$@"
