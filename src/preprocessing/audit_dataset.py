import cv2
import pandas as pd
import numpy as np
import os
import shutil
import hashlib

def print_dataset_dir(dataset_path):
    for dirname, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            print(os.path.join(dirname, filename))

def count_images(dataset_path):
    image_count = 0
    for dirname, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_count += 1
    return image_count

def count_labels(dataset_path):
    label_count = 0
    for dirname, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith('.txt'):
                label_count += 1
    return label_count

def class_distribution(dataset_path):
    pdf = pd.DataFrame(dataset_path)
    class_counts = pdf['class'].value_counts()
    return class_counts

def check_duplicate_images(dataset_path):
    image_hashes = {}
    duplicates = []
    for dirname, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(dirname, filename)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                if image_hash in image_hashes:
                    duplicates.append((image_path, image_hashes[image_hash]))
                else:
                    image_hashes[image_hash] = image_path
    return duplicates

def count_duplicates(dataset_path):
    duplicates = check_duplicate_images(dataset_path)
    return len(duplicates)



if __name__ == '__main__':
    dir = 'C:\\Users\\HP\\Desktop\\AGRIVISION\\dataset'
    # train_df = pd.read_csv(os.path.join(dir, 'Train.csv'))
    train_df = pd.read_csv('C:\\Users\\HP\\Desktop\\AGRIVISION\\Train.csv')
    # print_dataset_dir('C:\\Users\\HP\\Desktop\\AGRIVISION\\dataset')
    test_dir = 'C:\\Users\\HP\\Desktop\\AGRIVISION\\dataset\\images\\test'
    print(f'Total images: {count_images(dir)}')
    print(f'Total labels: {count_labels(dir)}')
    # print(f'Class distribution: {class_distribution(dir)}')
    print(f"Class distribution:\n{class_distribution(train_df)}")
    print("----"*10)
    print(f'Total Test Images: {count_images(test_dir)}')
    print(f'Image-Label Difference: {count_images(dir) - count_labels(dir)}')
    print("----"*10)
    print(f'Total duplicate images: {count_duplicates(dir)}')
    # print(f'Duplicate images: {check_duplicate_images(dir)}')
    print("----"*10)
    print("Dataset audit completed.")