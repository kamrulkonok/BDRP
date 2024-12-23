#!/bin/bash
#BATCH --job-name=cpy_file       # Job name
#SBATCH --output=cpy_file_%j.out # Output file
#SBATCH --error=cpy_file_%j.err  # Error file

#SBATCH --time=24:00:00              # Max runtime
#SBATCH --partition=gpua100          # Use the GPU A100 partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=4            # Number of CPUs per task
#SBATCH --mem=32G                    # Memory allocation
#SBATCH --ntasks=1                   # Number of tasks

# Run the Python script
python cpy_file.py
