"""
Fraud Detection System
Part 3: Model Building and Evaluation
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------

X = pd.read_pickle("output/X_train.pkl")
y = pd.read_pickle("output/y_train.pkl")

print(f"Samples: {len(X):,}, Features: {X.shape[1]}")
print(f"Fraud Rate: {y.mean() * 100:.2f}%")

# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print(f"Train Fraud: {pos:,}, Legitimate: {neg:,}")

# ------------------------------------------------------------
# Feature Scaling
# ------------------------------------------------------------

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# Model Definitions
# ------------------------------------------------------------

models = {
    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        "scaled": True
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),
        "scaled": False
    },

    "XGBoost": {
        "model": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=42
        ),
        "scaled": False
    },

    "LightGBM": {
        "model": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            is_unbalance=True,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ),
        "scaled": False
    },

    "CatBoost": {
        "model": CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=8,
            class_weights=[1, scale_pos_weight],
            verbose=0,
            random_state=42
        ),
        "scaled": False
    }
}

# ------------------------------------------------------------
# Training and Evaluation
# ------------------------------------------------------------

results = []
threshold = 0.25

for name, cfg in models.items():
    print(f"\nTraining {name}")

    if cfg["scaled"]:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    model = cfg["model"]
    model.fit(X_tr, y_train)

    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred),
        "ModelObject": model,
        "y_pred": y_pred
    }

    print(
        f"Accuracy={metrics['Accuracy']:.4f}, "
        f"Precision={metrics['Precision']:.4f}, "
        f"Recall={metrics['Recall']:.4f}, "
        f"F1={metrics['F1']:.4f}, "
        f"ROC_AUC={metrics['ROC_AUC']:.4f}"
    )

    results.append(metrics)

# ------------------------------------------------------------
# Model Comparison
# ------------------------------------------------------------

summary = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ["ConfusionMatrix", "ModelObject", "y_pred"]}
    for r in results
]).sort_values("F1", ascending=False)

print("\nModel Comparison:")
print(summary)

best_model_name = summary.iloc[0]["Model"]
best_result = next(r for r in results if r["Model"] == best_model_name)

print(f"\nBest Model: {best_model_name}")

# ------------------------------------------------------------
# Detailed Report
# ------------------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        best_result["y_pred"],
        target_names=["Legitimate", "Fraud"],
        digits=4
    )
)

# ------------------------------------------------------------
# Save Outputs
# ------------------------------------------------------------

joblib.dump(best_result["ModelObject"], "output/best_model.pkl")
joblib.dump(scaler, "output/scaler.pkl")
summary.to_csv("output/model_comparison.csv", index=False)

model_info = {
    "model": best_model_name,
    "threshold": threshold,
    "accuracy": best_result["Accuracy"],
    "precision": best_result["Precision"],
    "recall": best_result["Recall"],
    "f1": best_result["F1"],
    "roc_auc": best_result["ROC_AUC"],
    "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open("output/model_info.pkl", "wb") as f:
    pickle.dump(model_info, f)

print("\nModel training completed.")
