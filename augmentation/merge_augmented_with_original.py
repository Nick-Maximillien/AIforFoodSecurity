"""
merge_datasets.py

This script merges original and augmented YOLO-format datasets into a single
final dataset folder. It is useful after performing data augmentation where you
want to combine the original dataset with the augmented dataset for training.

---
WHAT IT DOES:
1. Defines paths for:
   - Original dataset (images + labels)
   - Augmented dataset (images + labels)
   - Final merged dataset (images + labels)

2. Creates the final dataset folders if they do not exist.

3. Copies all original images and labels into the final dataset.

4. Copies all augmented images and labels into the final dataset.

5. Prints a success message when done.

---
HOW TO RUN:
1. Place this script in the same directory where your `crop_data` folder is located.
   Folder structure should look like this:
       Desktop/crop_data/
           ├── images/
           ├── labels/
           ├── augmented/
           │   ├── images/
           │   └── labels/

2. Run the script from your terminal:
       python merge_datasets.py

3. The merged dataset will appear in:
       Desktop/crop_data/final_dataset/
           ├── images/
           └── labels/

4. Use this `final_dataset` folder for YOLO training.

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
