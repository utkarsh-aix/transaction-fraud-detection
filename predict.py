"""
Fraud Detection – Prediction Script
"""

import pandas as pd
import joblib
import pickle
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Load Model Artifacts
# ------------------------------------------------------------

MODEL_PATH = "output/best_model.pkl"
SCALER_PATH = "output/scaler.pkl"
MODEL_INFO_PATH = "output/model_info.pkl"
FEATURES_PATH = "output/feature_names.pkl"
ENCODERS_PATH = "output/label_encoders.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(MODEL_INFO_PATH, "rb") as f:
    model_info = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    feature_names = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    label_encoders = pickle.load(f)

THRESHOLD = model_info.get("threshold", 0.25)

print("Model loaded successfully")
print(f"Model: {model_info.get('model', 'Unknown')}")
print(f"Threshold: {THRESHOLD}")

# ------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------

def predict_fraud(data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict fraud for new transactions
    """

    df = data.copy()

    # Save TransactionID if present
    if "TransactionID" in df.columns:
        transaction_id = df["TransactionID"].values
        df = df.drop(columns=["TransactionID"])
    else:
        transaction_id = range(len(df))

    # Drop target if exists
    df = df.drop(columns=["isFraud"], errors="ignore")

    # Encode categorical columns
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            known = set(encoder.classes_)
            df[col] = df[col].apply(
                lambda x: encoder.transform([x])[0] if x in known else -999
            )

    # Ensure all required features exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = -999

    df = df[feature_names]
    df = df.fillna(-999)

    # Scale if required
    if model_info.get("uses_scaling", False):
        X = scaler.transform(df)
    else:
        X = df.values

    # Predict
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)

    results = pd.DataFrame({
        "TransactionID": transaction_id,
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
# Batch Prediction (Large Files)
# ------------------------------------------------------------

def predict_csv(input_csv, output_csv, chunk_size=10000):
    """
    Predict fraud for a large CSV file
    """

    results = []

    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        chunk_result = predict_fraud(chunk)
        results.append(chunk_result)

    final = pd.concat(results, ignore_index=True)
    final.to_csv(output_csv, index=False)

    return final

# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------

if __name__ == "__main__":

    try:
        sample = pd.read_pickle("output/X_train.pkl").sample(1000, random_state=42)
        preds = predict_fraud(sample)

        print("\nSample prediction summary:")
        print(preds["RiskLevel"].value_counts())

        preds.to_csv("output/sample_predictions.csv", index=False)
        print("Saved: output/sample_predictions.csv")

    except FileNotFoundError:
        print("No sample data found. Load your own CSV to predict.")
