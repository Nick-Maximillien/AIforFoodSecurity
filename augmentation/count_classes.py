"""
Class Distribution Counter for YOLO Labels
------------------------------------------

This script counts the number of labeled instances per class in a YOLO-format dataset.  
It goes through all label files in `crop_data/labels/`, extracts the class IDs, and tallies 
the occurrences. Then, it maps each class ID to its corresponding class name from 
`crop_data/classes.txt` and prints the results in a readable format.

Usage:
    - Place this script at the project root (where `crop_data/` folder is located).
    - Ensure `crop_data/labels/` contains YOLO-format `.txt` label files.
    - Ensure `crop_data/classes.txt` lists all class names, 0-indexed (one per line).
    - Run the script: `python class_counts.py`

Output:
    A list of classes and how many images/annotations exist for each, e.g.:

    0: Maize_Leaf_Blight → 432 images
    1: Tomato_Late_Blight → 275 images
    ...
"""

import os
from collections import defaultdict

# Path to YOLO labels folder
label_dir = 'crop_data/labels'

# Dictionary to store counts of each class
class_counts = defaultdict(int)

# Iterate over all label files
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):  # Only process label files
        with open(os.path.join(label_dir, label_file)) as f:
            lines = f.readlines()
            for line in lines:
                # Extract class ID (first value in each YOLO annotation line)
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

# Load class names (one per line, indexed 0..N-1)
with open('crop_data/classes.txt') as f:
    class_names = [line.strip() for line in f.readlines()]

# Print results sorted by class ID
for class_id, count in sorted(class_counts.items()):
    print(f"{class_id}: {class_names[class_id]} → {count} images")
