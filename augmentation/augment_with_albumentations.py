"""
augment_with_albumentations.py
---------------------------------
This script performs **data augmentation** for crop disease images using the
Albumentations library. It generates new synthetic images and bounding box
labels from a base dataset to improve training diversity.

📌 Key Features:
- Reads original images and YOLO-format labels (class + bbox).
- Applies a pipeline of augmentations (flip, crop, rotation, noise, weather).
- Saves augmented images and YOLO labels into a new `/augmented` folder.
- Augmentation targets (per class) are defined in `augment_plan.txt`.
- Each class is expanded up to 500 images for balanced training.

⚙️ Requirements:
- Python 3.8+
- albumentations
- OpenCV (cv2)
- A `crop_data` folder with structure:
    ├── images/       (original .jpg images)
    ├── labels/       (YOLO .txt labels)
    └── augment_plan.txt   (plan: class_id,base_filename)

💡 Output:
- Augmented images → `crop_data/augmented/images`
- Augmented labels → `crop_data/augmented/labels`

Usage:
    python augment_with_albumentations.py
"""

import os
import cv2
import random
import albumentations as A
from pathlib import Path

# --- CONFIG ---
ROOT = Path.home() / "Desktop" / "crop_data"
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"
AUG_IMAGES_DIR = ROOT / "augmented" / "images"
AUG_LABELS_DIR = ROOT / "augmented" / "labels"
PLAN_FILE = ROOT / "augment_plan.txt"

# Ensure output directories exist
AUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUG_LABELS_DIR.mkdir(parents=True, exist_ok=True)


# --- UTILITY FUNCTIONS ---
def yolo_to_bbox(cls_id, x, y, w, h, img_w, img_h):
    """
    Convert YOLO-format bbox → Pascal VOC (x_min, y_min, x_max, y_max).
    """
    x, y, w, h = float(x)*img_w, float(y)*img_h, float(w)*img_w, float(h)*img_h
    x_min = x - w/2
    y_min = y - h/2
    x_max = x + w/2
    y_max = y + h/2
    return [x_min, y_min, x_max, y_max]


def bbox_to_yolo(bbox, img_w, img_h):
    """
    Convert Pascal VOC bbox → YOLO format (x, y, w, h).
    Normalized to [0,1].
    """
    x_min, y_min, x_max, y_max = bbox
    x = ((x_min + x_max) / 2) / img_w
    y = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return [x, y, w, h]


# --- AUGMENTATION PIPELINE ---
transform = A.Compose([
    # Basic transformations
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=10, p=0.3),
    A.RandomCrop(height=256, width=256, p=0.2),
    A.GaussNoise(p=0.2),
    A.HueSaturationValue(p=0.3),
    A.MotionBlur(p=0.15),
    A.Affine(scale=(0.95, 1.05), translate_percent=0.05, rotate=(-10, 10), p=0.3),

    # Field realism (simulate lighting & weather)
    A.RandomShadow(p=0.2),
    A.RandomFog(p=0.15),
    A.RandomSunFlare(src_radius=30, flare_roi=(0.1, 0.1, 0.9, 0.3), p=0.1),

    # Background degradation
    A.ISONoise(p=0.2),
    A.Downscale(p=0.2),
    A.CLAHE(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


# --- LOAD AUGMENTATION PLAN ---
augment_targets = {}
with open(PLAN_FILE, "r") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        class_id, base = line.strip().split(",")
        class_id = int(class_id)
        augment_targets.setdefault(class_id, []).append(base)


# --- AUGMENTATION LOOP ---
for class_id, base_list in augment_targets.items():
    current_count = len(base_list)
    to_generate = 500 - current_count
    print(f"\n[CLASS {class_id}] Augmenting {to_generate} images")

    generated = 0
    attempts = 0
    while generated < to_generate:
        # Pick a random base image
        base = random.choice(base_list)
        img_path = IMAGES_DIR / f"{base}.jpg"
        label_path = LABELS_DIR / f"{base}.txt"

        if not img_path.exists() or not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        # Load YOLO bbox
        with open(label_path, "r") as f:
            cls_id, x, y, bw, bh = f.readline().strip().split()
        bbox = yolo_to_bbox(cls_id, x, y, bw, bh, w, h)

        try:
            # Apply augmentations
            aug = transform(image=image, bboxes=[bbox], class_labels=[cls_id])
            aug_img = aug["image"]
            aug_bbox = bbox_to_yolo(aug["bboxes"][0], aug_img.shape[1], aug_img.shape[0])

            # Save augmented image + label
            out_base = f"{base}_aug_{generated:03d}"
            cv2.imwrite(str(AUG_IMAGES_DIR / f"{out_base}.jpg"), aug_img)

            with open(AUG_LABELS_DIR / f"{out_base}.txt", "w") as f:
                f.write(f"{cls_id} {' '.join(f'{x:.6f}' for x in aug_bbox)}\n")

            generated += 1
            attempts = 0

        except Exception as e:
            attempts += 1
            if attempts > 100:
                print(f"⚠️ Skipping class {class_id}: too many failed attempts.")
                break

print("\n✅ DONE: Augmented images and labels saved to /augmented")
