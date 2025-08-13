# 🖋 Handwritten Digit Classification

## 📌 Project Overview
The project is about the classification of handwritten digits using machine learning and deep learning models.  
The dataset is from `sklearn.datasets.load_digits`, which contains 8x8 pixel grayscale images of digits 0–9.  

The notebook compares the performance of traditional ML algorithms and a Convolutional Neural Network (CNN).

## 🛠 Features
- Load and explore the digits dataset
- Preprocess data using MinMaxscaling
- Train and evaluate multiple models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Multi-layer Perceptron (MLP)
  - Convolutional Neural Network (CNN)
- Compare accuracy across all models
- Display classification reports

## 📂 Project Structure
│── src/ # Source code files
│── data/ # Data used 
│── notebook/ # Jupyter Notebook for experiments
│── requirements.txt # Python dependencies
│── README.md # Project documentation


## 📋 Requirements
Install the required dependencies:
Main libraries used:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow / keras

## 📊 Results

- notebook provides accuracy scores for all models and highlights the best-performing model