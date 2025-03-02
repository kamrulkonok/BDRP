#!/bin/bash
#SBATCH --job-name=deepcluster_test  # Job name
#SBATCH --output=slurm_scripts/deepcluster_test_%j.out  # Standard output file
#SBATCH --error=slurm_scripts/deepcluster_test_%j.err   # Standard error file
#SBATCH --time=24:00:00  # Shorter walltime for testing
#SBATCH --partition=gpu  # Use the test partition
#SBATCH --gres=gpu:4  # Request only 1 GPU for testing
#SBATCH --cpus-per-task=2  # Reduce CPU count for testing
#SBATCH --mem=256G  # Reduce memory for testing
#SBATCH --nodes=1  # Single node
#SBATCH --ntasks=1  # Single task

# Load required modules
module load cuda/11.7.0/gcc-11.2.0
module load python/3.9.10/gcc-11.2.0
module load anaconda3/2024.06/gcc-13.2.0
module load cudnn/8.6.0.163-11.8/oneapi-2023.2.1

# Activate Anaconda environment
source activate deepcluster

# Run Python script
python /gpfs/workdir/islamm/experiments/experiment_alexnet.py
