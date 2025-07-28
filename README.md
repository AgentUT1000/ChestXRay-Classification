# Chest X-Ray Disease Classification

A hybrid pipeline using OpenCV preprocessing + deep learning (ResNet50) to classify chest X-rays into multiple disease categories.

## Dataset
Curated dataset of 3710 chest X-ray images with multi-label annotations and metadata.

## Pipeline Components
- OpenCV preprocessing: resizing, histogram equalization, Gaussian noise, rotation
- Feature extraction: SIFT/ORB descriptors
- CNN fine-tuning: ResNet50 via transfer learning
- Grad-CAM visualization to interpret model focus
- Hybrid fusion of OpenCV features with CNN

## Visuals
Includes confusion matrices, ROC curves, and Grad-CAM overlays.

## Setup
```bash
pip install -r requirements.txt
