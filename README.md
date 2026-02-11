# 🐶🐱 Dogs vs Cats Image Classification using CNN

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify images of dogs and cats.  
The model is built using **TensorFlow/Keras** and trained on image data with multiple techniques applied to reduce overfitting and improve generalization.

The objective of this project is to accurately classify unseen images as:

- 🐶 Dog
- 🐱 Cat

---

## 🧠 Model Architecture

The CNN architecture consists of:

- 3 Convolutional Layers (32, 64, 128 filters)
- Batch Normalization after each Conv layer
- MaxPooling layers for dimensionality reduction
- Fully Connected (Dense) layers
- Dropout layers
- L2 Regularization
- Sigmoid activation in output layer (Binary Classification)

---

## ⚙️ Overfitting Reduction Techniques

To improve generalization and reduce overfitting, the following techniques were applied:

- ✅ Batch Normalization
- ✅ Dropout
- ✅ L2 Regularization
- ✅ Early Stopping
- ✅ Image Normalization ([0,255] → [0,1])

---

## 📊 Training Details

- **Image Size:** 256 × 256 × 3
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Evaluation Metric:** Accuracy
- **EarlyStopping:** Monitored on validation loss

### Initial Observation

- Training accuracy reached ~99%
- Validation accuracy remained around 70–75%
- Clear signs of overfitting were observed

### After Improvements

- Reduced overfitting
- Improved validation performance
- Better model generalization

---

## 🖼 Prediction Pipeline

To predict an unseen image:

1. Load image using OpenCV
2. Resize to (256, 256)
3. Normalize pixel values
4. Reshape to (1, 256, 256, 3)
5. Use `model.predict()`
6. Apply threshold (0.5) for final classification

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Google Colab

---

## 🚀 How to Run the Project

```bash
https://github.com/Devbrat-Singh/Image_Classification/blob/main/DogsVsCats_ImageClassification.ipynb
