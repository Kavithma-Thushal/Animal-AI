import os
import shutil
import kagglehub

# Step 1: Download the Animals-10 Dataset (Only if not already downloaded)
dataset_folder = "animals10_dataset"
if not os.path.exists(dataset_folder):
    print("Downloading Animals-10 dataset...")
    path = kagglehub.dataset_download("alessiocorrado99/animals10")
    shutil.move(path, dataset_folder)  # Move dataset to the desired folder
else:
    print("Dataset already exists. Skipping download.")

# Step 2: Define Paths
dataset_path = os.path.join(dataset_folder, "raw-img")

# Ensure dataset is properly extracted
if not os.path.exists(dataset_path):
    raise FileNotFoundError("Dataset folder not found. Check your extraction.")
