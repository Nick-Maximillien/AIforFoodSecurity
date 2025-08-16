"""
verify.py

This script merges original and augmented crop dataset files into a single
`final_dataset/` directory. It ensures that both images and corresponding YOLO
label files are combined into one clean dataset for training.

--------------------------------
WHAT THE SCRIPT DOES:
--------------------------------
1. Creates a `final_dataset/images` and `final_dataset/labels` directory.
2. Copies all original images (`*.jpg`) and YOLO labels (`*.txt`) into it.
3. Copies all augmented images and labels into the same folder.
4. Prints a success message when merging is complete.

--------------------------------
HOW TO RUN:
--------------------------------
1. Place this script in the same environment where your dataset folders exist.
   The expected folder structure is:

   crop_data/
   ├── images/        # original images
   ├── labels/        # original YOLO labels
   ├── augmented/
   │   ├── images/    # augmented images
   │   └── labels/    # augmented YOLO labels
   └── final_dataset/ # (this will be created automatically)

2. Open a terminal and run:
   python verify.py

3. After running, check the folder:
   crop_data/final_dataset/
   ├── images/  # contains both original + augmented images
   └── labels/  # contains both original + augmented labels
"""

import shutil
from pathlib import Path

# === PATHS ===
ROOT = Path.home() / "Desktop" / "crop_data"
ORIG_IMG = ROOT / "images"
ORIG_LBL = ROOT / "labels"
AUG_IMG = ROOT / "augmented" / "images"
AUG_LBL = ROOT / "augmented" / "labels"
FINAL_IMG = ROOT / "final_dataset" / "images"
FINAL_LBL = ROOT / "final_dataset" / "labels"

# === MAKE FINAL FOLDERS ===
FINAL_IMG.mkdir(parents=True, exist_ok=True)
FINAL_LBL.mkdir(parents=True, exist_ok=True)

# === COPY ORIGINAL ===
for file in ORIG_IMG.glob("*.jpg"):
    shutil.copy(file, FINAL_IMG / file.name)

for file in ORIG_LBL.glob("*.txt"):
    shutil.copy(file, FINAL_LBL / file.name)

# === COPY AUGMENTED ===
for file in AUG_IMG.glob("*.jpg"):
    shutil.copy(file, FINAL_IMG / file.name)

for file in AUG_LBL.glob("*.txt"):
    shutil.copy(file, FINAL_LBL / file.name)

print("\n✅ DONE: All original + augmented data merged to /final_dataset/")
