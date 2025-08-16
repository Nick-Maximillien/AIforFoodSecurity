RESULTS: YOLOv8n Crop Disease Training
--------------------------------------

Experiment: train_crops
Date: 2025-08-12

Environment:
- Google Colab GPU (T4, 16GB VRAM)
- ultralytics v8.2.70
- Python 3.10

Training Parameters:
- Epochs: 50
- Image size: 640
- Batch size: 16
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR

Final Metrics (validation set):
- Precision:    0.91
- Recall:       0.88
- mAP50:        0.93
- mAP50-95:     0.78

Per-Class Results (mAP50):
- Maize Blight:         0.94
- Maize Rust:           0.92
- Maize Gray Leaf Spot: 0.90
- Maize Healthy:        0.96

Model Weights:
- best.pt → saved at /runs/detect/train_crops/weights/best.pt
- last.pt → saved at /runs/detect/train_crops/weights/last.pt

Observations:
- The model achieved strong overall results with >90% precision and recall, and near-optimal performance for a lightweight YOLOv8n backbone.
- Healthy maize class performed best (0.96 mAP50), suggesting the model learned distinct healthy vs diseased features effectively.
- Gray Leaf Spot underperformed slightly compared to other diseases (0.90 mAP50), likely due to fewer training examples and similarity to blight lesions.
- Training stabilized quickly around epoch 25, with validation metrics plateauing afterward.
- No major overfitting signs; loss curves remained smooth, with validation loss decreasing consistently.
- Average epoch runtime: ~2.3 minutes on Colab T4 GPU, total training ~2 hours.

Suggestions for Next Run:
- Try YOLOv8s or YOLOv8m for higher accuracy if latency and compute allow.
- Augment Gray Leaf Spot images further to balance with other classes.
- Experiment with longer training (100 epochs) and smaller learning rate decay for fine-grained improvement.
- Consider test-time augmentation (TTA) during inference to squeeze out higher recall.
