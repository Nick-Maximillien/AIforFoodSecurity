"""
rescue_class.py
---------------------------------
This script "rescues" underrepresented classes in an object detection dataset
by generating synthetic samples using data augmentation. It ensures that the
chosen class has at least `TARGET_COUNT` samples available for training.

📌 Features:
- Reads original YOLO-format labels and identifies images containing the target class.
- Applies a pipeline of augmentations (flips, brightness/contrast, rotation, noise, weather, etc.).
- Saves augmented images and corresponding YOLO labels to a new `/augmented` folder.
- Continues generating augmented samples until the target class reaches `TARGET_COUNT`.

⚙️ Requirements:
- Python 3.8+
- albumentations
- OpenCV (cv2)
- Dataset structure:
    ├── images/       (original .jpg images)
    ├── labels/       (YOLO .txt labels)
    └── augmented/    (auto-created output folder)

💡 Usage:
    - Set `CLASS_ID` to the YOLO class index you want to rescue (string).
    - Set `TARGET_COUNT` to the desired number of samples for this class.
    - Run:
        python rescue_class.py
"""

import os
import cv2
import random
from pathlib import Path
import albumentations as A

# --- CONFIGURATION ---
ROOT = Path.home() / "Desktop" / "rescue_class"   # Root project folder
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"
AUG_DIR = ROOT / "augmented"
AUG_IMAGES_DIR = AUG_DIR / "images"
AUG_LABELS_DIR = AUG_DIR / "labels"

AUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUG_LABELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ID = "9"         # 👈 Change this to the class you want to rescue
TARGET_COUNT = 500     # 👈 Desired total sample count for this class


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
    A.Affine(scale=(0.95, 1.05), translate_percent=0.05,
             rotate=(-10, 10), p=0.3),

    # Field realism
    A.RandomShadow(p=0.2),
    A.RandomFog(p=0.15),
    A.RandomSunFlare(src_radius=30,
                     flare_roi=(0.1, 0.1, 0.9, 0.3),
                     p=0.1),

    # Background degradation
    A.ISONoise(p=0.2),
    A.Downscale(p=0.2),
    A.CLAHE(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc',
                            label_fields=['class_labels']))


# --- UTILITY FUNCTIONS ---
def yolo_to_bbox(cls_id, x, y, w, h, img_w, img_h):
    """
    Convert YOLO bbox (normalized center-x, center-y, w, h)
    → Pascal VOC format (x_min, y_min, x_max, y_max).
    """
    x, y, w, h = float(x)*img_w, float(y)*img_h, float(w)*img_w, float(h)*img_h
    x_min = x - w/2
    y_min = y - h/2
    x_max = x + w/2
    y_max = y + h/2
    return [x_min, y_min, x_max, y_max]


def bbox_to_yolo(bbox, img_w, img_h):
    """
    Convert Pascal VOC bbox → YOLO format (x, y, w, h),
    normalized to [0,1].
    """
    x_min, y_min, x_max, y_max = bbox
    x = ((x_min + x_max) / 2) / img_w
    y = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return [x, y, w, h]


# --- COLLECT ORIGINAL SAMPLES ---
all_bases = []
for label_file in LABELS_DIR.glob("*.txt"):
    with open(label_file, "r") as f:
        lines = f.readlines()
        # Keep only files that contain the target class
        if any(line.strip().startswith(CLASS_ID + " ") for line in lines):
            base = label_file.stem
            if (IMAGES_DIR / f"{base}.jpg").exists():
                all_bases.append(base)

original_count = len(all_bases)
to_generate = max(0, TARGET_COUNT - original_count)

print(f"\n🔢 Class {CLASS_ID}: Found {original_count} | "
      f"Need {to_generate} more samples")


# --- AUGMENTATION LOOP ---
generated = 0
attempts = 0
while generated < to_generate:
    base = random.choice(all_bases)
    img_path = IMAGES_DIR / f"{base}.jpg"
    label_path = LABELS_DIR / f"{base}.txt"

    image = cv2.imread(str(img_path))
    if image is None:
        continue
    h, w = image.shape[:2]

    # Extract only the target class bboxes
    with open(label_path, "r") as f:
        class_lines = [line.strip() for line in f
                       if line.startswith(CLASS_ID + " ")]
    if not class_lines:
        continue

    bboxes = []
    for line in class_lines:
        _, x, y, bw, bh = line.split()
        bboxes.append(yolo_to_bbox(CLASS_ID, x, y, bw, bh, w, h))

    try:
        # Apply augmentations
        aug = transform(image=image, bboxes=bboxes,
                        class_labels=[CLASS_ID]*len(bboxes))
        aug_img = aug["image"]

        # Save augmented image and label
        out_base = f"{base}_aug_{generated:03d}"
        cv2.imwrite(str(AUG_IMAGES_DIR / f"{out_base}.jpg"), aug_img)

        with open(AUG_LABELS_DIR / f"{out_base}.txt", "w") as f:
            for aug_box in aug["bboxes"]:
                aug_yolo = bbox_to_yolo(aug_box, aug_img.shape[1], aug_img.shape[0])
                f.write(f"{CLASS_ID} {' '.join(f'{x:.6f}' for x in aug_yolo)}\n")

        generated += 1
        attempts = 0

    except Exception as e:
        attempts += 1
        if attempts > 100:
            print(f"⚠️ Too many failed attempts. Skipping class {CLASS_ID}.")
            break

print(f"\n🎉 Done. Generated {generated} new samples for class {CLASS_ID} → {AUG_IMAGES_DIR}")
