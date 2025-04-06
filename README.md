# Medical Diagnosis AI

## Overview
This project utilizes deep learning and artificial intelligence to diagnose medical conditions, focusing on **Heart Disease and Pneumonia Detection** using X-ray images. The system aims to assist healthcare professionals by providing accurate disease classification.

## Features
- **Heart Disease and Pneumonia Detection** using Chest X-rays
- Image preprocessing and augmentation techniques
- Convolutional Neural Networks (CNNs) for image classification
- Performance evaluation using precision, recall, and F1-score
- Visualization of predictions and model confidence

## Technologies Used
- Python
- TensorFlow/Keras (for deep learning models)
- OpenCV (for image processing)
- Scikit-learn (for model evaluation)
- Pandas & NumPy (for data manipulation)
- Matplotlib & Seaborn (for data visualization)

## Dataset
The dataset consists of **Chest X-ray images** labeled for:
- **Pneumonia** (Bacterial/Viral)
- **Normal (Healthy Lungs)**
- **Heart Disease Indicators**

The dataset can be obtained from **public medical repositories** such as **Kaggle** or **NIH Chest X-ray Dataset**.

## Model Performance
- The system has been trained using **CNNs, ResNet, and Transformer-based models**.
- Achieved an **accuracy of ~90%** on benchmark datasets.

## Required Libraries
```python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
