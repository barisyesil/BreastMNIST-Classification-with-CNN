# BreastMNIST Classification with CNN

This project demonstrates the use of Convolutional Neural Networks (CNNs) to classify medical images from the **BreastMNIST** dataset, which is part of the MedMNIST collection. The model is built using **TensorFlow** and **Keras**, and it processes and classifies breast cancer images as benign or malignant.

---

## Table of Contents

- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

---

## Dataset

The dataset used in this project is **BreastMNIST**, a binary classification dataset containing images of breast tissue. The task is to classify the images into two categories:

- **Label 0**: Benign
- **Label 1**: Malignant

The images are of size **128x128 pixels** and are preprocessed for model training. The dataset is part of the [MedMNIST collection](https://medmnist.com/) and can be downloaded using the `medmnist` Python library.

---

## Problem Statement

The goal of this project is to develop and train a **Convolutional Neural Network (CNN)** to accurately classify breast cancer images from the **BreastMNIST** dataset. This project aims to explore deep learning techniques and apply them to medical image classification, which is critical in detecting early signs of breast cancer.

---

## Model Architecture

The model consists of the following layers:

1. **Convolutional Layers**: These layers are responsible for extracting features from the images using filters.
2. **MaxPooling Layers**: These layers reduce the spatial dimensions of the feature maps to reduce computation and avoid overfitting.
3. **Flatten Layer**: This layer flattens the 3D output from the convolutional layers into a 1D vector for input to the fully connected layers.
4. **Dense Layers**: These layers perform classification based on the extracted features.
5. **Dropout Layer**: This is a regularization technique that helps to prevent overfitting by randomly dropping a fraction of input units during training.
6. **Output Layer**: A single neuron with a **sigmoid** activation function for binary classification (benign or malignant).

The model is compiled using:

- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy

---

## Results

The model was trained for **10 epochs** with the following results:

- **Final Test Accuracy**: 77.56%

### Sample Predictions on Test Data:

- **Predicted: 1 | Actual: 0**
- **Predicted: 1 | Actual: 1**
- **Predicted: 1 | Actual: 1**
- **Predicted: 1 | Actual: 1**
- **Predicted: 1 | Actual: 1**

---

## License

Feel free to use these codes and the test results for your work.

---
