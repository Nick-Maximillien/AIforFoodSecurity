colab_training.md
# Colab Training Guide

This guide shows how to train your YOLOv8 crop disease model in **Google Colab**.  
The workflow mounts Google Drive, copies data locally for faster training, installs dependencies, and runs the training script.

---

## 1. Mount Google Drive

Mount your Google Drive so you can access your dataset and save checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')

2. Copy Dataset to Colab Environment

It’s faster to train from Colab’s local storage instead of streaming directly from Drive.

!mkdir -p /content/splits
!cp -r /content/drive/MyDrive/crop_data/train /content/splits/
!cp -r /content/drive/MyDrive/crop_data/val /content/splits/
!cp -r /content/drive/MyDrive/crop_data/test /content/splits/

3. Install Dependencies

Install the Ultralytics package:

!pip install ultralytics --upgrade

4. Check GPU Availability

Verify that CUDA (GPU) is available:

import torch
torch.cuda.is_available()


If it returns True, you’re good to go 🚀

5. Create data.yaml

YOLO requires a dataset configuration file. Update this with your dataset paths and class list.

data_yaml = """
train: /content/crop_data/train/images
val: /content/crop_data/val/images
test: /content/crop_data/test/images

names:
  0: Maize Blight
  1: Maize Rust
  2: Maize Gray Leaf Spot
  3: Maize Healthy
"""

with open('/content/crop_data/data.yaml', 'w') as f:
    f.write(data_yaml)

6. Train YOLOv8 Model

Run training with the Ultralytics YOLO API:

from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train
model.train(
    data='/content/crop_data/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    workers=2,
    name='train_crops'
)

7. Save Outputs

After training, all logs and model weights are saved under:

/content/runs/detect/train_crops/


Copy them back to Drive:

!cp -r /content/runs/detect/train_crops /content/drive/MyDrive/
