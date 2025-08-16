"""
split_dataset.py

This script splits your final YOLO dataset into **train**, **valid**, and **test**
subsets. It also generates a `data.yaml` file required for YOLOv8 training.

--------------------------------
WHAT THE SCRIPT DOES:
--------------------------------
1. Reads all image/label pairs from `crop_data/final_datasets/`.
2. Ensures only valid pairs are used (skips any image without a label).
3. Randomly splits the dataset into train/valid/test subsets according to ratios.
4. Copies images and labels into:
   - crop_data/splits/train/
   - crop_data/splits/valid/
   - crop_data/splits/test/
5. Generates a `data.yaml` inside `crop_data/splits/` with:
   - Dataset paths
   - Class names loaded from `classes.txt`

--------------------------------
HOW TO RUN:
--------------------------------
1. Make sure you have the following structure on your Desktop:

   crop_data/
   ├── classes.txt
   ├── final_datasets/
   │   ├── images/   # all merged images
   │   └── labels/   # all merged YOLO labels
   └── splits/       # will be created automatically

2. Adjust the split ratios (TRAIN_RATIO, VALID_RATIO, TEST_RATIO) if needed.

3. Run in terminal:
   python split_dataset.py

4. After running, check:
   crop_data/splits/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   ├── test/
   └── data.yaml   # config file for YOLOv8
"""

import os
import shutil
import random
from pathlib import Path
import yaml

# --- CONFIG ---
ROOT = Path.home() / "Desktop"
FINAL_DATASET_DIR = ROOT / "crop_data" / "final_datasets"   # Input dataset
SPLITS_ROOT = ROOT / "crop_data" / "splits"                # Output splits

TRAIN_DIR = SPLITS_ROOT / "train"
VALID_DIR = SPLITS_ROOT / "valid"
TEST_DIR = SPLITS_ROOT / "test"  # Optional for YOLOv8
CLASSES_FILE = FINAL_DATASET_DIR.parent / "classes.txt"    # ~/Desktop/crop_data/classes.txt

# Split ratios (train/valid/test must sum to 1.0)
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

# --- UTILITY ---
def copy_files(file_list, src_dir, dest_dir):
    """
    Copies a list of files from source directory to destination directory.
    Creates the destination directory if it doesn't exist.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for file in file_list:
        src = src_dir / file
        dest = dest_dir / file
        if src.exists():
            shutil.copy(src, dest)
        else:
            print(f"⚠️ Missing file: {src}")

# --- PREPARE DIRECTORIES ---
for split_dir in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)

# --- LOAD DATA ---
image_dir = FINAL_DATASET_DIR / "images"
label_dir = FINAL_DATASET_DIR / "labels"

image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

# --- FILTER VALID IMAGE/LABEL PAIRS ---
valid_image_files = []
for f in image_files:
    label = f.replace(".jpg", ".txt")
    if (label_dir / label).exists():
        valid_image_files.append(f)
    else:
        print(f"⚠️ No label for: {f}")

# --- SPLIT INTO TRAIN/VALID/TEST ---
random.shuffle(valid_image_files)

train_count = int(len(valid_image_files) * TRAIN_RATIO)
valid_count = int(len(valid_image_files) * VALID_RATIO)
test_count = len(valid_image_files) - train_count - valid_count  # remainder

train_files = valid_image_files[:train_count]
valid_files = valid_image_files[train_count:train_count + valid_count]
test_files = valid_image_files[train_count + valid_count:]

# --- COPY FILES TO SPLITS ---
for files, split in [
    (train_files, TRAIN_DIR),
    (valid_files, VALID_DIR),
    (test_files, TEST_DIR),
]:
    print(f"📦 Copying {len(files)} samples to {split}")
    # Copy images
    copy_files(files, image_dir, split / "images")
    # Copy corresponding labels
    copy_files([f.replace(".jpg", ".txt") for f in files], label_dir, split / "labels")

# --- WRITE data.yaml FOR YOLOv8 ---
with open(CLASSES_FILE, 'r') as f:
    class_list = [line.strip() for line in f if line.strip()]

yaml_data = {
    "path": str(SPLITS_ROOT),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",  # Optional, YOLOv8 supports test split
    "names": {i: name for i, name in enumerate(class_list)}
}

yaml_path = SPLITS_ROOT / "data.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f, default_flow_style=False)

print(f"\n✅ All done. data.yaml saved to:\n{yaml_path}")
