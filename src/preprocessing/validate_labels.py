import os
import pandas as pd
import cv2
from pathlib import Path


def validate_labels(csv_path, images_dir,labels_dir):
    """Validate that each image in the CSV file has a corresponding label file and that the labels are correctly formatted.
    Args:
        csv_path (str): Path to the CSV file containing image paths and labels.
        images_dir (str): Directory where the images are stored.
    Returns:
        bool: True if all labels are valid, False otherwise.
    """
    df = pd.read_csv(csv_path)
    valid = True

    for index, row in df.iterrows():
        image_path = images_dir / Path(row['Image_ID'])
        label_path = labels_dir / Path(row['Image_ID']).with_suffix('.txt')
        # label_path = image_path.with_suffix('.txt')

        # Check if the image file exists
        # label_path = image_path.replace('.jpg', '.txt')

        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            valid = False
            continue

        # Check if the label file exists
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            valid = False
            continue

        # Validate the label format (e.g., check if it contains valid class IDs and bounding box coordinates)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Invalid label format in {label_path}: {line.strip()}")
                    valid = False
                    break
                class_id, x_center, y_center, width, height = parts
                if not class_id.isdigit() or int(class_id) < 0:
                    print(f"Invalid class ID in {label_path}: {class_id}")
                    valid = False
                    break
                try:
                    float(x_center)
                    float(y_center)
                    float(width)
                    float(height)
                except ValueError:
                    print(f"Invalid bounding box coordinates in {label_path}: {line.strip()}")
                    valid = False
                    break

    return valid

if __name__ == "__main__":
    csv_path = 'C://Users//HP//Desktop//AGRIVISION//New_Train.csv'
    images_dir = 'C://Users//HP//Desktop//AGRIVISION//dataset//images//train'
    labels_dir = 'C://Users//HP//Desktop//AGRIVISION//dataset//labels//train'
    val_labels_dir = 'C://Users//HP//Desktop//AGRIVISION//dataset//labels//val'
    val_csv = 'C://Users//HP//Desktop//AGRIVISION//New_Valid.csv'
    
    if validate_labels(csv_path=csv_path, images_dir=images_dir, labels_dir=labels_dir):
        print("All Training labels are valid.")
    else:
        print("Some Tranining labels are invalid. Please check the output for details.")
    # validate_labels(csv_path, images_dir, val_labels_dir)
    
    if validate_labels(csv_path=val_csv, images_dir=images_dir, labels_dir=val_labels_dir):
        print("All validation labels are valid.")
    else:
        print("Some validation labels are invalid. Please check the output for details.")