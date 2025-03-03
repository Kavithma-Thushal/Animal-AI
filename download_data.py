import os
import shutil
import kagglehub

dataset_folder = "animals10_dataset"
if not os.path.exists(dataset_folder):
    print("Downloading dataset...")
    path = kagglehub.dataset_download("alessiocorrado99/animals10")
    shutil.move(path, dataset_folder)
else:
    print("Dataset already exists")
