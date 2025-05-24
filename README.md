
# 🛡️ Online Fraud Detection

This project aims to detect fraudulent transactions in an online payment system using machine learning techniques. It includes thorough preprocessing, exploratory data analysis (EDA), and model training to identify suspicious activities in real-time or batch settings.

## 📁 Project Overview

The notebook performs the following steps:

* Load and explore the dataset
* Handle class imbalance
* Visualize class distribution and key features
* Apply machine learning models to detect fraud
* Evaluate model performance with appropriate metrics

## 🧠 ML Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* XGBoost (if included)
* Support Vector Machine (SVM)

> Class imbalance techniques such as **SMOTE** or **undersampling** might be applied if present in the notebook.

## 📊 Dataset Description

Includes features like:

* `Transaction Amount`
* `Time`
* `Transaction Type`
* `Old Balance` vs `New Balance`
* Binary target: `isFraud` (1 for fraud, 0 for legitimate)

## 📌 Key Features

* **Data Cleaning**: Handling missing values and inconsistent data.
* **EDA**: Distribution plots, correlation heatmaps, outlier analysis.
* **Class Imbalance Handling**: Ensures better fraud detection.
* **Model Evaluation**:

  * Confusion Matrix
  * Precision, Recall, F1-score
  * ROC-AUC Score

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Imbalanced-learn (if SMOTE/ADASYN used)

## 📍 Insights

* Fraudulent transactions are rare and need special handling.
* Certain transaction types or value patterns may indicate higher fraud risk.
* Model performance is highly dependent on how well class imbalance is addressed.

## 🚀 How to Use

1. Clone/download the repository or the `.ipynb` file.
2. Open the notebook in Jupyter Notebook or Google Colab.
3. Run the cells step-by-step to understand or modify the pipeline.
4. Evaluate fraud detection performance or integrate with your own transaction dataset.

## 🔐 Use Cases

* Fraud detection in fintech or e-commerce platforms
* Financial anomaly detection systems
* Credit card fraud classification
* Transaction monitoring systems

---
