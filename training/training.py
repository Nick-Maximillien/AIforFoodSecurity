# COLAB TRAINING SCRIPT
# This script is designed to be run in Google Colab for training a model.
# It sets up the environment, installs necessary packages, and runs the training script.

# Make sure to run this in a Google Colab environment.

# Mount Google Drive to access files
# This allows you to save and load files directly from your Google Drive.
from google.colab import drive
drive.mount('/content/drive')

# copy training data from Google Drive to the Colab environment
# Saves time by avoiding scanning directly from Google Drive during training
!mkdir -p /content/splits
!cp -r /content/drive/MyDrive/crop_data/train /content/splits/
!cp -r /content/drive/MyDrive/crop_data/val /content/splits/
!cp -r /content/drive/MyDrive/crop_data/test /content/splits/


# Install the ultralytics package
# This package is required for training the model.
!pip install ultralytics --upgrade


# Check if CUDA is available
# This is important for utilizing GPU acceleration during training.
import torch
torch.cuda.is_available()


# Point to the data.yaml file
# This file contains the dataset configuration, including paths and class names.
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


# Run the training script using the ultralytics package
# This will start the training process using the specified dataset and parameters.
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
