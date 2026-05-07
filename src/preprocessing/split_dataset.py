import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
# import multiprocessing

# Set the data directory
DATA_DIR = Path('C://Users//HP//Desktop//AGRIVISION//dataset')
CSV_DIR = Path("C://Users//HP//Desktop//AGRIVISION//")

# Load the CSV files
train = pd.read_csv(CSV_DIR / 'Train.csv')
test = pd.read_csv(CSV_DIR / 'Test.csv')
ss = pd.read_csv(CSV_DIR / 'SampleSubmission.csv')


def split_dataset(train_csv, output_dir):
    # Load the CSV files
    train = pd.read_csv(train_csv)
    # test = pd.read_csv(test_csv)

    # Map class labels to numeric values
    train['class_id'] = train['class'].map(
        {'healthy': 0, 'anthracnose': 1, 'cssvd': 2})

    # Split data into training and validation sets based on unique Image_IDs to avoid data leakage
    train_unique_imgs_df = train.drop_duplicates(subset=['Image_ID'], ignore_index=True)
    X_train, X_val = train_test_split(train_unique_imgs_df, test_size=0.30, stratify=train_unique_imgs_df['class'], random_state=42)

    X_train.to_csv(output_dir / "New_Train.csv", index=False)
    X_val.to_csv(output_dir / "New_Valid.csv", index=False)

    # Remove the validation images from the training set
    X_train = train[train.Image_ID.isin(X_train.Image_ID)]
    X_val = train[train.Image_ID.isin(X_val.Image_ID)]

    return X_train, X_val

def extract_and_copy_files(X_val, images_dir, val_images_dir ):
    if not val_images_dir.exists():
        val_images_dir.mkdir(parents=True, exist_ok=True)
    for img in tqdm(X_val.Image_ID.unique()):
        print("Splitting Validation dataset")
        shutil.copy(images_dir / img, val_images_dir / img)
        # img_ext = img.split('.')[-1]
        # label_name = img.replace(f'.{img_ext}', '.txt')
        # shutil.copy(labels_dir / label_name, val_labels_dir / label_name)
        print("Copied", img, "to validation set")   


if __name__ == "__main__":
    # Define paths
    train_csv = CSV_DIR / 'Train.csv'
    output_dir = CSV_DIR

    # Split dataset
    X_train, X_val = split_dataset(train_csv, output_dir)

    # Define directories for images and labels
    TRAIN_IMAGES_DIR = DATA_DIR / "images" / 'train'
    TRAIN_LABELS_DIR = DATA_DIR  / 'train' / 'labels'
    VAL_IMAGES_DIR = DATA_DIR / 'train' / 'val'
    VAL_LABELS_DIR = DATA_DIR / 'train' / 'val' / 'labels'

    # Extract and copy files to validation set
    extract_and_copy_files(X_val, TRAIN_IMAGES_DIR,  VAL_IMAGES_DIR)
    
    # Delete Created validation set
    shutil.rmtree(VAL_IMAGES_DIR)
    