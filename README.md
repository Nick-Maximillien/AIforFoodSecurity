# Developer README — Agrosight AI

This document provides a **technical overview** of how the Agrosight AI crop diagnosis system was developed, including dataset sourcing, annotation, augmentation, training, evaluation, and deployment. It is meant for researchers, collaborators, and contributors to understand the development process.  
⚠️ Note: Production code and private stack details are **not disclosed** here.

---

## 1. Dataset Collection

- **Source:**  
  - ~80% of data was sourced from **Medley datasets** (Ghana and Burkina Faso).  
  - ~20% was **locally collected** by the Agrosight team.  

- **Target Crops & Classes:**  
  - The **main AI engine** currently supports **28 classes** across maize, tomato, rice, and tea.  
  - This repo open-sources the **first maize dataset** (4 maize classes) for learning and research purposes.

---

## 2. Data Labeling

- Tool: **LabelImg** (YOLO format)  
- Process:  
  - Annotated images one by one for bounding boxes.  
  - Classes maintained in `classes.txt` (0-indexed).  
  - Stored under `labels/` directory.

---

## 3. Data Augmentation

- Tools: **Albumentations + OpenCV**  
- Purpose: Balance dataset and enrich model generalization.  
- Applied techniques:  
  - Rotation, flipping, brightness/contrast adjustment, blurring, noise addition.  
  - All classes brought to **500+ images each**.  

Scripts are stored under:  
```bash
augmentation/
    augment.py
    utils.py
4. Training
Platform: Google Colab (GPU)

Model: YOLOv8

Classes: Initially trained on 4 maize classes, later extended to 28 full classes.

Scripts under training/:

training/
    train.py
    val.py
    dataset.yaml
5. Evaluation
Metrics logged: mAP, precision, recall, F1.

Validation runs are under metrics/.

Best performing weights stored in models/best.pt.

6. Deployment
Multi-channel access:

Web app (Next.js): Agrosight Detect

WhatsApp Uploads: Agrosight WhatsApp Detect

Mobile APK: (Android package build of the web app)

/observe Dashboard: For monitoring and farmer data insights.

https://agrosight-ai.vercel.app/observe


7. Funding & Monetization Strategy
Enterprise AI Expansion:
Training models for sorting and grading automation in large-scale agribusiness.

Partnerships:
With fertilizer and treatment product makers/sellers to promote products via recommendations.

8. Directory Structure

Agrosight-AI/
│
├── README.md                # Public overview
├── developer_readme.md      # This file (full dev process)
│
├── crop_data/               # Maize dataset (open-sourced 4 classes)
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── augmentation/            # Augmentation scripts
│   ├── augment_with_albumentatiions.py
│   └── prepare_dataset.py and other augmentation scripts
│
├── training/                # Training & validation scripts
│   ├── training.py
│  
│
├── metrics/                 # Evaluation results
│   ├── results.txt
│   └── confusion_matrix.png
│
├── models/                  # Trained weights
│   └── best.pt
│
└── docs/                    # Documentation
    ├── augmentation.md
    └── colab_training.md

9. Notes for Researchers

The open-sourced maize dataset (4 classes) and best.pt are provided for educational and research purposes only.

For full dataset access (28 classes), please contact the Agrosight team.

Contributions are welcome for data collection, labeling, and regional expansion.

10. Live Links
🌍 Web deployment:
 https://agrosight-ai.vercel.app/web

📱 WhatsApp Deployment:
https://agrosight-ai.vercel.app/whatsapp

📝 Agrosight AI page:
https://www.notion.so/Agrosight-AI-23cfd0d4350d80d9a25dccef402872d3?source=copy_link

🚜 Pilot Page:
https://www.notion.so/Agrosight-AI-Pilot-Strategy-Climate-Smart-Crop-Diagnostics-251fd0d4350d806681c0efb6714bcfa2?source=copy_link
