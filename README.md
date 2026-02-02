# Sports Celebrity Image Classification 

A **Machine Learning image classification project** that identifies sports celebrities from images using computer vision techniques and supervised learning models.

---

## Web App Screenshot

![Sports Celebrity Classification Web App](ui_snapshot.jpg)

> Add the website screenshot inside a `screenshots/` folder.

---

## Problem Statement
Given an input image of a sports celebrity, classify it into the correct celebrity category.

---

## Machine Learning Pipeline

### Data Processing
- Collected images class-wise (celebrity folders)
- Face and upper-body detection using OpenCV
- Image resizing and normalization

### Feature Engineering
- Wavelet Transform for texture extraction
- Combined raw pixel features with wavelet features

### Model Training
- Algorithms tested: SVM, Logistic Regression
- Best model selected using GridSearchCV

### Output
- Predicted sports celebrity label

---

## Model Details
- **Type:** Multi-class classification  
- **Input:** Processed image features  
- **Output:** Celebrity name  

The trained model and feature columns are saved using `pickle` and `JSON`.

---

## Deployment
- Flask backend serving predictions
- REST API for image-based classification
- Simple web interface for uploading images

---

## Tech Stack
Python · OpenCV · NumPy · Scikit-learn · Flask · Jupyter Notebook

---

## About
An end-to-end ML classification system demonstrating data preprocessing, feature engineering, model evaluation, and deployment.

---

## Author
**Aanchal Routela**
