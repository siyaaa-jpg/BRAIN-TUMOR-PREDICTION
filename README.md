# Brain Tumor Detection using CNN

This repository contains a deep learning model for detecting brain tumors using Convolutional Neural Networks (CNN). The model is trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), achieving high accuracy on both training and testing sets.

## Dataset
The dataset used in this project is sourced from Kaggle and contains MRI images categorized into two classes: **Tumor** and **No Tumor**. You can access the dataset [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

## Model Overview
The model is built using a Convolutional Neural Network (CNN), which is highly effective in image classification tasks. The architecture of the CNN consists of multiple convolutional layers, pooling layers, and fully connected layers, designed to extract features and classify MRI images.

### Model Performance:
- **Training Accuracy**: 98.99%
- **Testing Accuracy**: 96.99%

## Dependencies
- Python 3.x
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

To install the dependencies, run:
```bash
pip install -r requirements.txt

