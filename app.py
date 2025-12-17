import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

# ------------------------------------------------------------
# Load Model Artifacts (cached)
# ------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("output/best_model.pkl")
    scaler = joblib.load("output/scaler.pkl")

    with open("output/model_info.pkl", "rb") as f:
        model_info = pickle.load(f)

    with open("output/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    with open("output/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    return model, scaler, model_info, feature_names, label_encoders


model, scaler, model_info, feature_names, label_encoders = load_artifacts()
DEFAULT_THRESHOLD = model_info.get("threshold", 0.25)

# ------------------------------------------------------------
# Helper: preprocess & predict
# ------------------------------------------------------------

def preprocess_and_predict(df, threshold=DEFAULT_THRESHOLD):
    data = df.copy()

    # Handle TransactionID
    if "TransactionID" in data.columns:
        txn_id = data["TransactionID"]
        data = data.drop(columns=["TransactionID"])
    else:
        txn_id = range(len(data))

    # Drop target if present
    data = data.drop(columns=["isFraud"], errors="ignore")

    # Encode categorical columns
    for col, encoder in label_encoders.items():
        if col in data.columns:
            data[col] = data[col].astype(str)
            known = set(encoder.classes_)
            data[col] = data[col].apply(
                lambda x: encoder.transform([x])[0] if x in known else -999
            )

    # Ensure feature alignment
    for col in feature_names:
        if col not in data.columns:
            data[col] = -999

    data = data[feature_names].fillna(-999)

    # Scale if required
    if model_info.get("uses_scaling", False):
        X = scaler.transform(data)
    else:
        X = data.values

    # Predict
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    results = pd.DataFrame({
        "TransactionID": txn_id,
        "Prediction": pred,
        "FraudProbability": prob
    })

    results["RiskLevel"] = pd.cut(
        prob,
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["Low", "Medium", "High", "Critical"]
    )

    return results

# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------

st.sidebar.title("Fraud Detection")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Single Transaction",
        "Batch Prediction",
        "Risk Analysis",
        "Model Performance"
    ]
)

# ------------------------------------------------------------
# HOME
# ------------------------------------------------------------

if page == "Home":
    st.title("Transaction Fraud Detection System")

    st.write(
        "This application detects fraudulent financial transactions using a "
        "machine learning model. The system focuses on high recall to reduce "
        "missed fraud cases."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Recall", f"{model_info['recall']:.2f}")
    col2.metric("Precision", f"{model_info['precision']:.2f}")
    col3.metric("ROC-AUC", f"{model_info['roc_auc']:.2f}")

    st.subheader("Why this matters")
    st.write(
        "Fraud detection is a highly imbalanced problem. Accuracy alone is not "
        "reliable, so this system prioritizes recall, precision, and "
        "probability-based risk scoring."
    )

# ------------------------------------------------------------
# SINGLE TRANSACTION
# ------------------------------------------------------------

elif page == "Single Transaction":
    st.title("Single Transaction Prediction")

    st.write("Enter transaction details to predict fraud risk.")

    input_data = {}
    for col in feature_names[:10]:  # limit inputs for simplicity
        input_data[col] = st.text_input(col)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        result = preprocess_and_predict(input_df)

        prob = result.loc[0, "FraudProbability"]
        label = "Fraud" if result.loc[0, "Prediction"] == 1 else "Legitimate"
        risk = result.loc[0, "RiskLevel"]

        st.subheader("Prediction Result")
        st.write(f"Label: **{label}**")
        st.write(f"Fraud Probability: **{prob:.3f}**")
        st.write(f"Risk Level: **{risk}**")

# ------------------------------------------------------------
# BATCH PREDICTION
# ------------------------------------------------------------

elif page == "Batch Prediction":
    st.title("Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview")
        st.dataframe(data.head())

        if st.button("Run Prediction"):
            result = preprocess_and_predict(data)

            fraud_count = (result["Prediction"] == 1).sum()

            st.subheader("Summary")
            st.write(f"Total Transactions: {len(result)}")
            st.write(f"Fraud Detected: {fraud_count}")

            st.dataframe(result.head(20))

            st.download_button(
                "Download Predictions",
                result.to_csv(index=False),
                file_name="fraud_predictions.csv"
            )

# ------------------------------------------------------------
# RISK ANALYSIS
# ------------------------------------------------------------

elif page == "Risk Analysis":
    st.title("Risk Analysis")

    uploaded_file = st.file_uploader(
        "Upload CSV file for risk analysis",
        type=["csv"]
    )

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        threshold = st.slider(
            "Fraud Probability Threshold",
            min_value=0.1,
            max_value=0.9,
            value=DEFAULT_THRESHOLD,
            step=0.05
        )

        result = preprocess_and_predict(data, threshold)

        st.subheader("Risk Distribution")
        st.bar_chart(result["RiskLevel"].value_counts())

        high_risk = result[result["RiskLevel"].isin(["High", "Critical"])]
        st.subheader("High Risk Transactions")
        st.dataframe(high_risk.head(20))

# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------

elif page == "Model Performance":
    st.title("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{model_info['accuracy']:.2f}")
    col2.metric("Precision", f"{model_info['precision']:.2f}")
    col3.metric("Recall", f"{model_info['recall']:.2f}")

    col4, col5 = st.columns(2)
    col4.metric("F1-Score", f"{model_info['f1']:.2f}")
    col5.metric("ROC-AUC", f"{model_info['roc_auc']:.2f}")
