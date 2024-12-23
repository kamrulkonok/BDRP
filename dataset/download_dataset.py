import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Define the target directory for the dataset
target_dir = "/gpfs/workdir/islamm/datasets"
os.makedirs(target_dir, exist_ok=True)

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset
try:
    print("Starting dataset download...")
    api.dataset_download_files(
        "nih-chest-xrays/data", path=target_dir, unzip=True
    )
    print(f"Dataset downloaded to: {target_dir}")
except Exception as e:
    print(f"Error downloading dataset: {str(e)}")

