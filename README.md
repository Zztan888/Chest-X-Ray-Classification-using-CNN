# 🩻 Chest X-Ray Classification using CNN

This repository features a Deep Learning project focused on medical imaging. We implement a **Convolutional Neural Network (CNN)** to classify chest X-ray images, specifically aimed at detecting abnormalities such as Pneumonia.

---

## 📑 Project Components
* **Model Architecture:** Convolutional Neural Network (CNN)
* **Dataset:** [Pneumonia X-Ray Images](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images/data) (Images of Normal vs. Pneumonia cases)
* **Language/Framework:** Python, TensorFlow/Keras or PyTorch
* **Deliverables:** Jupyter Notebook and Technical Report

---

## 📂 Quick Links
* 📓 **Source Code (Jupyter):** [Assignment_4_CNN.ipynb](./Assignment_4_CNN.ipynb)
* 📄 **Technical Report:** [Report_Assignment_4.pdf](./Report_Assignment_4-git.pdf)
* 📁 **Dataset Folder:** [Pneumonia X-Ray Images](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images/data)

## 📊 Dataset Information
The dataset used in this project is sourced from Kaggle. It contains thousands of pediatric chest X-ray images categorized into **Normal** and **Pneumonia**.

**To use the dataset with this code:**
1. Download the data from: [Kaggle - Pneumonia X-Ray Images](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images/data)
2. Extract the contents.
3. Ensure your folder structure looks like this:
   ```text
   /your-repo-folder
   ├── Assignment_4_CNN.ipynb
   └── chest_xray/
       ├── train/
       ├── test/
       └── val/

---

## 🧠 Introduction to CNNs
A **Convolutional Neural Network (CNN)** is a class of deep neural networks most commonly applied to analyzing visual imagery. Unlike standard neural networks, CNNs use "filters" to automatically learn spatial hierarchies of features from input images.



### How it works in this project:
1. **Convolutional Layers:** Apply filters to the X-ray to detect edges, shapes, and lung opacities.
2. **Pooling (Downsampling):** Reduces the dimensionality of the data while keeping the most important features.
3. **Activation (ReLU):** Introduces non-linearity to the model.
4. **Fully Connected Layer:** Classifies the features into categories (e.g., **Normal** vs. **Pneumonia**).

---

## 💻 Implementation Highlights
The notebook `Assignment_4_CNN.ipynb` includes:
* **Data Augmentation:** Techniques like rotation and zooming to prevent overfitting and improve model robustness.
* **Model Training:** Using the Adam optimizer and Categorical Cross-Entropy loss.
  $$Loss = -\sum_{i=1}^{c} y_i \log(\hat{y}_i)$$
* **Evaluation:** Using Confusion Matrices and Accuracy/Loss curves to track performance.



---

## 📈 Results & Analysis
Detailed findings and performance metrics are documented in the [Assignment Report](./Report_Assignment_4-git.pdf).
* **Accuracy:** The model's ability to correctly identify Pneumonia cases.
* **Precision/Recall:** Critical metrics in medical fields to minimize false negatives.

---

## 🚀 How to Run
1. Ensure the `chest_xray` dataset is placed in the root directory.
2. Install the required libraries:
   ```bash
   pip install tensorflow matplotlib pandas numpy
