"""
Augmentation Preparation Script
-------------------------------

This script prepares an **augmentation plan** for underrepresented classes 
in a YOLO dataset. It identifies images that contain only a single class 
(single-class images are safer for augmentation) and selects those belonging 
to target classes that need more samples. The result is saved in a 
`augment_plan.txt` file, which can later be used by an augmentation script 
to generate additional training data.

Workflow:
1. Load class names from `classes.txt`.
2. Parse YOLO label files to identify images containing only one class.
3. Keep only the classes listed in `needed_counts` (the classes that require augmentation).
4. Display how many valid images were found vs. how many are needed.
5. Save an `augment_plan.txt` file mapping class IDs to base filenames.

Usage:
    - Place this script in the dataset root (where `crop_data/` exists).
    - Update `needed_counts` with the shortfall of images per class.
    - Run: `python prepare_augmentation.py`
    - Output: `crop_data/augment_plan.txt` (list of candidate images for augmentation)

"""

import os
from pathlib import Path
from collections import defaultdict

# --- CONFIG ---
ROOT = Path.home() / "Desktop" / "crop_data"
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"
CLASSES_FILE = ROOT / "classes.txt"

# Output folders (prepared for augmented data later)
OUTPUT_DIR = ROOT / "augmented"
AUG_IMAGES_DIR = OUTPUT_DIR / "images"
AUG_LABELS_DIR = OUTPUT_DIR / "labels"
AUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUG_LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Target dataset size (per class)
TARGET_COUNT = 500  

# Current class shortfalls (how many more are needed to reach 500)
needed_counts = {
    6: 13, 9: 228, 10: 410, 12: 184, 13: 375,
    14: 437, 18: 171, 19: 156, 20: 177, 21: 191,
    22: 198, 23: 191, 24: 201, 25: 113, 26: 152, 27: 21
}

# --- STEP 1: Load class names ---
with open(CLASSES_FILE, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# --- STEP 2: Parse YOLO label files to find single-class images ---
single_class_images = defaultdict(list)

for label_path in LABELS_DIR.glob("*.txt"):
    with open(label_path, "r") as f:
        lines = f.readlines()
    # Collect all class IDs in this label file
    classes = [line.strip().split()[0] for line in lines]
    
    if len(set(classes)) == 1:  # only one unique class present
        class_id = int(classes[0])
        if class_id in needed_counts:  # only track underrepresented classes
            base = label_path.stem  # file name without extension
            img_path = IMAGES_DIR / f"{base}.jpg"
            if img_path.exists():
                single_class_images[class_id].append(base)

# --- STEP 3: Display statistics ---
for class_id, base_list in single_class_images.items():
    current = len(base_list)
    needed = needed_counts[class_id]
    print(f"Class {class_id:02d} ({class_names[class_id]}): {current} valid → Need {needed}")

# --- STEP 4: Build augmentation plan ---
augment_plan = {}
for class_id, base_list in single_class_images.items():
    needed = needed_counts[class_id]
    if needed > 0:
        augment_plan[class_id] = base_list

# --- STEP 5: Save plan for audit/augmentation ---
with open(ROOT / "augment_plan.txt", "w") as f:
    for class_id, base_list in augment_plan.items():
        f.write(f"# Class {class_id} ({class_names[class_id]})\n")
        for base in base_list:
            f.write(f"{class_id},{base}\n")

print("\n✅ Done. Ready for augmentation. Check 'augment_plan.txt'")
