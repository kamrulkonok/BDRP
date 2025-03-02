#!/bin/bash
#SBATCH --job-name=deepcluster_se2_exp2_test   # Changed job name for testing
#SBATCH --output=slurm_scripts/deepcluster_exp2_se2_test_%j.out  # Output file
#SBATCH --error=slurm_scripts/deepcluster_exp2_se2_test_%j.err   # Error file
#SBATCH --time=00:30:00                   # Reduced walltime for testing
#SBATCH --partition=gpu              # Use the GPU test partition
#SBATCH --gres=gpu:4                      # Request 4 GPUs
#SBATCH --cpus-per-task=4                 # Request 4 CPUs
#SBATCH --mem=256G                        # Reduced memory for testing
#SBATCH --nodes=1                         # Run on a single node
#SBATCH --ntasks=1                        # Single task managing all GPUs

# Load required modules
module load cuda/11.7.0/gcc-11.2.0
module load python/3.9.10/gcc-11.2.0
module load anaconda3/2024.06/gcc-13.2.0
module load cudnn/8.6.0.163-11.8/oneapi-2023.2.1

# Activate your Anaconda environment
source activate deepcluster

# Run the SE(2) training Python script
python /gpfs/workdir/islamm/experiments/experiment2_se2.py
