"""
Script: filter_class.py

Description:
------------
This script filters out all images and label files belonging to a specific YOLO class
from a dataset and copies them into a new folder for analysis, augmentation, or training.

How it works:
-------------
1. Iterates through YOLO-format label files (.txt).
2. Checks if the file contains the target class ID.
3. If found, copies both the label file and its corresponding image (.jpg) 
   into a separate output folder.
4. Prints the total number of matched samples.

Use cases:
----------
- Extracting a single class for debugging or inspection.
- Creating a balanced subset of data for augmentation or testing.
- Cleaning or isolating data during dataset preparation.

Author: Nicholas Muthoki (Agrosight AI)
"""

import os
import shutil

# Paths: adjust these to your dataset structure
label_dir = r"C:/Users/hp/Desktop/crop_data/labels"      # Folder with YOLO label files
image_dir = r"C:/Users/hp/Desktop/crop_data/images"      # Folder with corresponding images
output_dir = r"C:/Users/hp/Desktop/filtered_class_12"    # Output folder for filtered data

# Create output folders if they don't exist
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

# Set the class to filter (string, since YOLO class IDs are written as strings in label files)
target_class = "12"
count = 0  # Track number of files copied

# Iterate over label files
for label_file in os.listdir(label_dir):
    if label_file.endswith(".txt"):
        label_path = os.path.join(label_dir, label_file)

        # Read label file
        with open(label_path, "r") as f:
            lines = f.readlines()

            # Check if any line in this file starts with the target class ID
            if any(line.startswith(target_class + " ") for line in lines):
                # Copy label file
                shutil.copy(label_path, os.path.join(output_dir, "labels", label_file))

                # Copy corresponding image file (assumes .jpg format)
                image_file = label_file.replace(".txt", ".jpg")
                shutil.copy(os.path.join(image_dir, image_file),
                            os.path.join(output_dir, "images", image_file))

                count += 1

print(f"Copied {count} images and labels for class {target_class}")
