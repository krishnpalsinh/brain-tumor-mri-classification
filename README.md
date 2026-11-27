# Explainable Brain Tumor MRI Classification (DenseNet121 + Grad-CAM)

An end-to-end deep learning computer vision pipeline that classifies brain MRI scans into four categories (**Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**) and generates **Grad-CAM visual heatmaps** to highlight the exact tumor regions driving model predictions.

## Key Features & Architecture
- **Backbone:** Pre-trained **DenseNet121** (ImageNet weights) adapted for medical grayscale/RGB MRI scans.
- **Two-Stage Transfer Learning:** 
  1. Trained a regularized classification head (`GlobalAveragePooling2D` + `Dropout(0.3)`).
  2. Fine-tuned the top 60 convolutional layers with `BatchNormalization` layers locked and `ReduceLROnPlateau` scheduling to preserve spatial edge features.
- **Data Augmentation:** Real-time random horizontal flips, rotations, and zooms to prevent overfitting.
- **Explainable AI (Grad-CAM):** Extracts gradients from DenseNet121's final convolutional activation (`relu`) to overlay attention heatmaps on uploaded MRI scans.
- **Interactive Web UI:** Built a live drag-and-drop prediction app using **Gradio**.

## Model Performance
- **Dataset:** Brain Tumor MRI Dataset (Kaggle — 7,023 total scans across 4 classes).
- **Test Accuracy:** **[90.44%]** on unseen test images.
- **Evaluation:** Assessed via per-class Precision, Recall, F1-score, and Confusion Matrix analysis.

## Tech Stack
- **Deep Learning:** Python, TensorFlow / Keras
- **Computer Vision & Visualization:** OpenCV, Matplotlib, Seaborn, PIL
- **Deployment / UI:** Gradio, Google Colab (T4 GPU)

## How to Run
1. Open `Brain_Tumor_MRI_Classifier.ipynb` using the **Open in Colab** badge at the top of the notebook.
2. Enable GPU acceleration (`Runtime > Change runtime type > T4 GPU`).
3. Run all cells to download the dataset via `kagglehub`, train/evaluate the model, and launch the Gradio + Grad-CAM interface.
