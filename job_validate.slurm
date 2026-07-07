#!/bin/bash
#SBATCH --job-name=val_chem
#SBATCH --output=validate_%j.out
#SBATCH --error=validate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=4G

# The script takes arguments passed to sbatch
# Example: sbatch job_validate.slurm --use_rdkit

python3 validate.py "$@"
