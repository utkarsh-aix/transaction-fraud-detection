# 💳 Transaction Fraud Detection System

An end-to-end **Machine Learning based Transaction Fraud Detection system** that detects fraudulent transactions using advanced classification models and provides a **Streamlit web application** for real-time and batch predictions.

---

## 📌 Problem Statement

Financial fraud is a major challenge in digital transactions. Fraudulent activities often represent a **very small percentage of total transactions**, making detection difficult due to **high class imbalance**.

Traditional accuracy-based models fail to identify fraud effectively. Therefore, this project focuses on **recall, precision, and probability-based risk scoring** to minimize fraud losses.

---

## 🎯 Project Objectives

- Detect fraudulent transactions with high recall
- Handle highly imbalanced transaction data
- Build multiple ML models and select the best one
- Provide real-time and batch prediction support
- Deliver a clean and professional Streamlit application

---

## 🧠 Project Description

This project implements a complete fraud detection pipeline:

- Feature engineering on transaction data
- Training multiple ML models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost)
- Handling class imbalance using model-based weighting
- Model evaluation using Recall, Precision, F1-score, ROC-AUC
- Probability-based fraud risk classification
- Deployment-ready Streamlit web application

---

## 🗂️ Project Structure

transaction-fraud-detection/
│
├── app.py # Streamlit application
├── predict.py # Prediction logic
├── model_building.py # Model training script
├── create_test_dataset.py # Synthetic test data generator
│
├── output/
│ ├── best_model.pkl
│ ├── scaler.pkl
│ ├── model_info.pkl
│ ├── feature_names.pkl
│ └── label_encoders.pkl
│
├── requirements.txt
└── README.md


---

## ⚙️ Technologies Used

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **XGBoost, LightGBM, CatBoost**
- **Streamlit**
- **Joblib**

---

## 🚀 How to Run the Application

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt


### Run the Streamlit App
streamlit run app.py


🧪 Testing the Application
Batch Prediction

Upload test_transactions.csv

View fraud predictions

Download results

Risk Analysis

Adjust probability threshold

Analyze high-risk transactions

Observe risk distribution

📊 Model Performance (Example)

Recall: ~87%

Precision: ~19%

ROC-AUC: ~0.94

High recall ensures most fraud cases are detected, even at the cost of false positives.

📈 Application Features

🏠 Home: Model overview & metrics

🧾 Single Transaction Prediction

📂 Batch CSV Prediction

⚠️ Risk Analysis with threshold tuning

📊 Model Performance Dashboard

👤 Author

Utkarsh Raj
M.Tech – Applied Data Science & Artificial Intelligence
