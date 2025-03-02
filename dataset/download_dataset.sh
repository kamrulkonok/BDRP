#!/bin/bash
#SBATCH --job-name=kaggle_download    # Job name
#SBATCH --output=kaggle_download_%j.out # Output file
#SBATCH --error=kaggle_download_%j.err  # Error file
#SBATCH --time=01:00:00               # Adjusted max runtime
#SBATCH --partition=cpu_short         # Use CPU_SHORT partition
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # Number of CPUs per task
#SBATCH --mem=32G                     # Memory


# Export Kaggle API credentials as environment variables
export KAGGLE_USERNAME="kamrulislamkonok"
export KAGGLE_KEY="3ef8ed2121b8fda5d15fe9da89c6db1a"


# Run the Python script to download the dataset
python3 download_dataset.py

