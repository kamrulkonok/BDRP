#!/bin/bash
#SBATCH --job-name=deepcluster_p4m_exp2   # Changed job name
#SBATCH --output=slurm_scripts/deepcluster_exp2_p4m_%j.out  # Output file saved in slurm_scripts folder
#SBATCH --error=slurm_scripts/deepcluster_exp2_p4m_%j.err   # Error file saved in slurm_scripts folder
#SBATCH --time=24:00:00                   # Maximum execution time
#SBATCH --partition=gpu               # Use the GPU partition
#SBATCH --gres=gpu:4                      # Request 4 GPUs
#SBATCH --cpus-per-task=4                 # Request 4 CPUs
#SBATCH --mem=256G                        # Request 256GB RAM
#SBATCH --nodes=1                         # Run on a single node
#SBATCH --ntasks=1                        # Run a single task

# Load required modules
module load cuda/11.7.0/gcc-11.2.0
module load python/3.9.10/gcc-11.2.0
module load anaconda3/2024.06/gcc-13.2.0
module load cudnn/8.6.0.163-11.8/oneapi-2023.2.1

# Activate your Anaconda environment
source activate deepcluster

# Run the SE(2) training Python script
python /gpfs/workdir/islamm/experiments/experiment2_p4m.py