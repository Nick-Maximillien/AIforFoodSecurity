# Agrosight AI — Data Preparation & Augmentation Guide

This guide documents the **complete dataset preparation workflow** for Agrosight AI.  
It consolidates all the scripts we have built into a single pipeline.

---

## 1. Prepare Data
Raw collected crop disease data should be organized in the following structure:

crop_data/
├── images/ # Raw images (JPEG)
├── labels/ # YOLO-format labels (.txt)
├── classes.txt # List of all class names (1 per line)



Each YOLO label file has the format:
<class_id> <x_center> <y_center> <width> <height>


All values except `<class_id>` are normalized to `[0,1]`.

---

## 2. Count Classes

Script: **count_classes.py**

📌 Purpose:  
Counts how many samples exist for each class in `labels/`. Helps identify class imbalance.

🚀 Usage:
```bash
python count_classes.py
✅ Output:

Total images

Class distribution report

3. Verify Dataset Integrity
Script: verify_dataset_integrity.py

📌 Purpose:
Ensures images and labels match correctly and that class IDs are valid.

Checks performed:

Images with no label

Labels with no image

Invalid or out-of-range class IDs

Class frequency distribution

🚀 Usage:


python verify_dataset_integrity.py
✅ Output:

Warnings for mismatches

Example invalid files

Per-class label distribution

4. Run Augmentation
Script: augment_with_albumentations.py

📌 Purpose:
Expands dataset by simulating real farm conditions (fog, blur, low light, glare, shadows, etc.) using Albumentations.
Generates augmented images + updated YOLO annotations.

🛠 Workflow:

Reads augmentation targets from augment_plan.txt

Format: <class_id>,<image_base_name>

Example:


0,maize_blight_001
1,rust_leaf_010
Loads corresponding images + YOLO labels.

Applies random transformations:

Flip, rotate, blur, noise

Weather effects (fog, shadows, sun flare)

Color/contrast changes

Saves to:


crop_data/augmented/images/
crop_data/augmented/labels/
⚙️ Config:

Target: ~500 images per class

Input: crop_data/images/, labels/

Output: crop_data/augmented/

🚀 Usage:


python augment_with_albumentations.py
✅ Output:

Augmented dataset (~500 images/class)

Logs of generation progress

5. Merge Datasets
After augmentation, merge original + augmented sets:

crop_data/
├── final_datasets/
│   ├── images/
│   └── labels/
Simply copy or script-merge images/ + labels/ from both original and augmented/.

6. Split Dataset
Script: split_dataset.py

📌 Purpose:
Splits dataset into train/valid/test sets for YOLOv8 and generates data.yaml.

🛠 Workflow:

Random shuffle of all valid image/label pairs

70% → train, 20% → valid, 10% → test (default)

Copies into:

bash
Copy
Edit
crop_data/splits/
├── train/images, train/labels
├── valid/images, valid/labels
├── test/images, test/labels
Writes data.yaml with paths + class names

🚀 Usage:

python split_dataset.py
✅ Output:

YOLO-ready dataset splits

data.yaml in crop_data/splits/

Final Notes
Always run verify_dataset_integrity.py before augmentation to catch errors early.

Use count_classes.py after merging to check balance.

Adjust split ratios in split_dataset.py if needed.

YOLOv8 training command (example):


yolo detect train data=crop_data/splits/data.yaml model=yolov8n.pt epochs=100 imgsz=640
👨‍💻 Author: Nicholas Muthoki
Project: Agrosight AI — Data Augmentation Pipeline