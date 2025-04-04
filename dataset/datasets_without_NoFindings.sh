#!/bin/bash
#SBATCH --job-name=create_dataset_without_Nofindings       # Job name
#SBATCH --output=dataset_without_Nofindings_%j.out # Output file
#SBATCH --error=_dataset_without_Nofindings_%j.err  # Error file

#SBATCH --time=04:00:00              # Max runtime
#SBATCH --partition=gpua100          # Use the GPU A100 partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=4            # Number of CPUs per task
#SBATCH --mem=32G                    # Memory allocation
#SBATCH --ntasks=1                   # Number of tasks

# Run the Python script
python datasets_without_NoFindings.py
