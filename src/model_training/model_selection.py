import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import torch

train = pd.read_csv('C://Users//HP//Desktop//AGRIVISION//New_Train.csv')
# Create a data.yaml file required by YOLO
class_names = train['class'].unique().tolist()
num_classes = len(class_names)
print('num_classes',num_classes)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device model is on",device)

# TRAIN_IMAGES_DIR =
TRAIN_IMAGES_DIR = Path("C:/Users/HP/Desktop/AGRIVISION/dataset/images/train")
VAL_IMAGES_DIR = Path("C:/Users/HP/Desktop/AGRIVISION/dataset/images/val")
# LABEL_DIR = Path("C:/Users/HP/Desktop/AGRIVISION/dataset/labels/val")



data_yaml = {
    'train': str(TRAIN_IMAGES_DIR),
    'val': str(VAL_IMAGES_DIR),
    'nc': num_classes,
    'names': class_names
}

# Save the data.yaml file
yaml_path = Path('data.yaml')
with open(yaml_path, 'w') as file:
    yaml.dump(data_yaml, file, default_flow_style=False)


def select_model(model_path,data_yaml : str,device : str,epochs=20,img_size=640):

    # Load a YOLO pretrained model
    model = YOLO(model_path).to(device)
    # print(model)

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data=data_yaml,  # Path to dataset configuration file
        epochs=epochs,  # Number of training epochs
        imgsz=img_size,  # Image size for training
        device=device,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

    # Plot the training results
    model.plot_results(train_results)

    print("Training completed!",train_results)
    print("Training completed!")
    print("---"*20)


if __name__ == "__main__":
    select_model(model_path="yolov8s.pt",data_yaml='data.yaml',device=device,epochs=10,img_size=640)