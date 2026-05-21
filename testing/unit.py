from src.preprocessing import split_dataset, visualize_annotation
from src.model_training import model_selection
from pathlib import Path
import pandas as pd
from src.preprocessing.split_dataset import extract_and_copy_files
from src.preprocessing.visualize_annotation import load_annotations, plot_image_with_boxes
from src.model_training.model_selection import select_model, create_data_yaml
import os
import torch

# Set the data directory
DATA_DIR = Path('C://Users//HP//Desktop//AGRIVISION//dataset')
CSV_DIR = Path("C://Users//HP//Desktop//AGRIVISION//")

# Load the CSV files
train = pd.read_csv(CSV_DIR / 'Train.csv')
test = pd.read_csv(CSV_DIR / 'Test.csv')
ss = pd.read_csv(CSV_DIR / 'SampleSubmission.csv')

    # Define paths
train_csv = CSV_DIR / 'Train.csv'
output_dir = CSV_DIR

train_csv = train_csv[:5]
# Split dataset
X_train, X_val = split_dataset(train_csv, output_dir)

# Define directories for images and labels
TRAIN_IMAGES_DIR = DATA_DIR / "images" / 'train'
TRAIN_LABELS_DIR = DATA_DIR  / 'train' / 'labels'
VAL_IMAGES_DIR = DATA_DIR / 'images' / 'val'
VAL_LABELS_DIR = DATA_DIR / 'images' / 'val' / 'labels'

# Extract and copy files to validation set
extract_and_copy_files(X_val, TRAIN_IMAGES_DIR,  VAL_IMAGES_DIR)

for image_name in os.listdir(TRAIN_IMAGES_DIR)[499:505]:
        # print(image_name)
        image_path = TRAIN_IMAGES_DIR / Path(image_name)
        # print(image_path)
        img_ext = image_name.split('.')[-1]
        label_path = TRAIN_LABELS_DIR / Path(image_name.replace(f'.{img_ext}', '.txt'))
        print(label_path)

        if label_path.exists():
            boxes = load_annotations(label_path)
            print(f"Plotting {image_name} with {len(boxes)} bounding boxes.")
            plot_image_with_boxes(image_path, boxes)
        else:
            print(f"No annotations found for {image_name}.")
            
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device model is on",device)

class_names = train['class'].unique().tolist()
num_classes = len(class_names)
print('num_classes',num_classes)

data = create_data_yaml(train_path=train_csv, val_path=None, class_names=class_names, output_path=CSV_DIR / 'data.yaml')

# Define paths for model and data
select_model(model_path="yolov26s.pt",data_yaml= data,
                device=device,epochs=10,
                img_size=640,project='yolov26s')