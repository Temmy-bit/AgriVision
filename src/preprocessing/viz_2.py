import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path


# Load YOLO annotations
def load_annotations(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    boxes = []

    for line in lines:
        class_id, x_center, y_center, width, height = map(
            float,
            line.strip().split()
        )

        boxes.append((class_id, x_center, y_center, width, height))

    return boxes


# Draw bounding boxes
def draw_boxes(ax, image_path, boxes):

    image = np.array(Image.open(str(image_path)))

    h, w, _ = image.shape

    ax.imshow(image)

    for box in boxes:

        class_id, x_center, y_center, width, height = box

        # YOLO -> pixel coordinates
        xmin = int((x_center - width / 2) * w)
        ymin = int((y_center - height / 2) * h)
        xmax = int((x_center + width / 2) * w)
        ymax = int((y_center + height / 2) * h)

        # Rectangle
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            edgecolor='red',
            facecolor='none',
            linewidth=2
        )

        ax.add_patch(rect)

        # Label
        ax.text(
            xmin,
            ymin - 5,
            f'Class {int(class_id)}',
            color='yellow',
            fontsize=8,
            weight='bold',
            backgroundcolor='black'
        )

    ax.set_title(Path(image_path).name, fontsize=8)
    ax.axis('off')


if __name__ == "__main__":

    IMAGE_DIR = Path("C:/Users/HP/Desktop/AGRIVISION/dataset/images/val")
    LABEL_DIR = Path("C:/Users/HP/Desktop/AGRIVISION/dataset/labels/val")

    # Get all image files
    image_files = [
        img for img in os.listdir(IMAGE_DIR)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    # Randomly sample images
    num_images = 6
    random_images = random.sample(
        image_files,
        min(num_images, len(image_files))
    )

    # Grid layout
    rows = 2
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    for ax, image_name in zip(axes, random_images):

        image_path = IMAGE_DIR / image_name

        img_ext = image_name.split('.')[-1]

        label_path = LABEL_DIR / image_name.replace(
            f'.{img_ext}',
            '.txt'
        )

        if label_path.exists():

            boxes = load_annotations(label_path)

            print(f"{image_name}: {len(boxes)} boxes")

            draw_boxes(ax, image_path, boxes)

        else:

            image = np.array(Image.open(str(image_path)))

            ax.imshow(image)

            ax.set_title(f"{image_name}\nNo Labels")

            ax.axis('off')

    # Hide unused subplot spaces
    for ax in axes[len(random_images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()