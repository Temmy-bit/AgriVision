import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Plot some images and their bboxes to ensure the conversion was done correctly
def load_annotations(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes

# Function to plot an image with its bounding boxes
def plot_image_with_boxes(image_path, boxes):
    # Load the image
    image = np.array(Image.open(str(image_path)))


    # Get image dimensions
    h, w, _ = image.shape

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Plot each bounding box
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        # Convert YOLO format to corner coordinates
        xmin = int((x_center - width / 2) * w)
        ymin = int((y_center - height / 2) * h)
        xmax = int((x_center + width / 2) * w)
        ymax = int((y_center + height / 2) * h)

        # Draw the bounding box
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                          edgecolor='red', facecolor='none', linewidth=2))
        plt.text(xmin, ymin - 10, f'Class {int(class_id)}', color='red', fontsize=8, weight='bold')

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    
    csv_path = 'C://Users//HP//Desktop//AGRIVISION//New_Train.csv'
    IMAGE_DIR = 'C://Users//HP//Desktop//AGRIVISION//dataset//images//train'
    LABEL_DIR = 'C://Users//HP//Desktop//AGRIVISION//dataset//labels//train'
    val_labels_dir = 'C://Users//HP//Desktop//AGRIVISION//dataset//labels//val'
    val_csv = 'C://Users//HP//Desktop//AGRIVISION//New_Valid.csv'

    # Plot a few images with their annotations
    for image_name in os.listdir(IMAGE_DIR)[499:505]:
        # print(image_name)
        image_path = IMAGE_DIR / Path(image_name)
        # print(image_path)
        img_ext = image_name.split('.')[-1]
        label_path = LABEL_DIR / Path(image_name.replace(f'.{img_ext}', '.txt'))
        print(label_path)

        if label_path.exists():
            boxes = load_annotations(label_path)
            print(f"Plotting {image_name} with {len(boxes)} bounding boxes.")
            plot_image_with_boxes(image_path, boxes)
        else:
            print(f"No annotations found for {image_name}.")
