from pathlib import Path
import cv2
import multiprocessing
from tqdm import tqdm
import pandas as pd

# Function to convert the bounding boxes to YOLO format and save them
def save_yolo_annotation(row):
    image_path, class_id, output_dir = row['ImagePath'], row['class_id'], row['output_dir']

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    height, width, _ = img.shape
    label_file = Path(output_dir) / f"{Path(image_path).stem}.txt"

    ymin, xmin, ymax, xmax = row['ymin'], row['xmin'], row['ymax'], row['xmax']

    # Normalize the coordinates
    x_center = (xmin + xmax) / 2 / width
    y_center = (ymin + ymax) / 2 / height
    bbox_width = (xmax - xmin) / width
    bbox_height = (ymax - ymin) / height

    with open(label_file, 'a') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# Parallelize the annotation saving process
def process_dataset(dataframe, output_dir):
    dataframe['output_dir'] = output_dir
    with multiprocessing.Pool() as pool:
        list(tqdm(pool.imap(save_yolo_annotation, dataframe.to_dict('records')), total=len(dataframe)))

# Save train and validation labels to their respective dirs
# process_dataset(X_train, TRAIN_LABELS_DIR)
# process_dataset(X_val, VAL_LABELS_DIR)


if __name__ == "__main__":

    DATA_DIR = Path('C://Users//HP//Desktop//AGRIVISION//dataset')
    # Load the train and validation dataframes
    X_train = pd.read_csv("New_Train.csv")
    X_val = pd.read_csv("New_Valid.csv")

    TRAIN_LABELS_DIR = DATA_DIR / 'labels' / 'train'
    VAL_LABELS_DIR = DATA_DIR / 'labels' / 'val'
    process_dataset(X_train, TRAIN_LABELS_DIR)
    process_dataset(X_val, VAL_LABELS_DIR)


