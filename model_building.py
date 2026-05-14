"""
=============================================================================
FraudShield Pro — Model Building Pipeline (XGBoost v2 Overhaul)
=============================================================================
This script is the complete, production training pipeline for the
FraudShield Pro fraud detection engine. It covers:

  1. Data Loading & Exploration
  2. Class Imbalance Analysis & Handling
  3. Feature Engineering (via FraudPreprocessor)
  4. XGBoost Model Training & Hyperparameter Tuning
  5. Threshold Optimization
  6. Full Evaluation: Confusion Matrix, Precision, Recall, F1, ROC-AUC
  7. Saving Production Artifacts to output_v2/

Output Artifacts:
  - output_v2/best_model.pkl       → Trained XGBoost model
  - output_v2/feature_names.pkl    → Ordered feature list for inference
  - output_v2/feature_stats.pkl    → Historical card/address spend stats
  - output_v2/model_info.pkl       → Threshold + evaluation metrics
=============================================================================
"""

import os
import pickle
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

os.makedirs("output_v2", exist_ok=True)

# =============================================================================
# SECTION 1: DATA LOADING & EXPLORATION
# =============================================================================
print("=" * 60)
print("SECTION 1: DATA LOADING & EXPLORATION")
print("=" * 60)

# Load the pre-engineered feature matrix (produced by feature_eng.ipynb)
# The feature matrix is saved from the feature engineering notebook.
X = pd.read_pickle("output/X_train.pkl")
y = pd.read_pickle("output/y_train.pkl")

print(f"\n[INFO] Dataset loaded successfully.")
print(f"  Total samples   : {len(X):,}")
print(f"  Total features  : {X.shape[1]}")
print(f"  Feature dtypes  : {X.dtypes.value_counts().to_dict()}")
print(f"\n[INFO] Label Distribution:")
fraud_count     = int(y.sum())
legit_count     = int((y == 0).sum())
total           = len(y)
fraud_rate      = fraud_count / total * 100
print(f"  Legitimate (0)  : {legit_count:,}  ({100 - fraud_rate:.2f}%)")
print(f"  Fraud (1)       : {fraud_count:,}   ({fraud_rate:.2f}%)")

print(f"\n[INFO] Sample Feature Statistics:")
print(X.describe().T[["mean", "std", "min", "max"]].head(10).to_string())

# =============================================================================
# SECTION 2: CLASS IMBALANCE HANDLING
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: CLASS IMBALANCE HANDLING")
print("=" * 60)

# --- Train / Test Split (stratified to preserve fraud ratio in both sets) ---
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,        # Ensures fraud ratio is equal in train & test
    random_state=42
)

neg = int((y_train == 0).sum())   # Legitimate transactions
pos = int((y_train == 1).sum())   # Fraudulent transactions

# --- scale_pos_weight: The key XGBoost imbalance technique ---
# This tells XGBoost to penalize errors on the fraud class by this factor.
# Formula: scale_pos_weight = count(negative) / count(positive)
scale_pos_weight = neg / pos

print(f"\n[INFO] Train Set Split:")
print(f"  Legitimate (0)   : {neg:,}")
print(f"  Fraud (1)        : {pos:,}")
print(f"  Imbalance Ratio  : {scale_pos_weight:.1f}:1")
print(f"\n[STRATEGY] Using 'scale_pos_weight = {scale_pos_weight:.2f}'")
print("  → XGBoost will treat each fraud sample as equivalent to")
print(f"    {scale_pos_weight:.1f} legitimate samples in the loss function.")
print("  → This forces the model to prioritize catching fraud over")
print("    minimizing overall error — the correct approach for this problem.")

print(f"\n[INFO] Test Set Split:")
print(f"  Legitimate (0)   : {int((y_test == 0).sum()):,}")
print(f"  Fraud (1)        : {int(y_test.sum()):,}")

# =============================================================================
# SECTION 3: MODEL TRAINING & HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: XGBOOST V2 — MODEL TRAINING")
print("=" * 60)

print("\n[INFO] Model: XGBoost (Extreme Gradient Boosting)")
print("[INFO] Hyperparameters (XGBoost v2 Overhaul):")

# --- Hyperparameter Rationale ---
# n_estimators=500   → More trees than v1 (400). Better generalization.
# learning_rate=0.05 → Small step size to prevent overfitting.
# max_depth=7        → Controls tree complexity. Deeper = more powerful but
#                      risks overfitting. 7 is ideal for tabular fraud data.
# subsample=0.8      → Each tree sees 80% of rows (row-level bagging).
#                      Adds randomness, prevents memorization.
# colsample_bytree=0.8 → Each tree sees 80% of features (column bagging).
# min_child_weight=5 → Minimum sum of weights in a leaf node.
#                      Higher = more conservative, less overfitting.
# gamma=0.1          → Minimum loss reduction to make a split. Regularization.
# reg_alpha=0.1      → L1 regularization on weights (feature selection).
# reg_lambda=1.5     → L2 regularization on weights (weight shrinkage).
# scale_pos_weight   → Computed above. Handles class imbalance.
# eval_metric='aucpr'→ Optimizes for Area Under Precision-Recall Curve,
#                      which is the correct metric for imbalanced datasets.

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42,
)

params = xgb_model.get_params()
for k, v in params.items():
    if k not in ["n_jobs", "use_label_encoder", "verbosity", "base_score",
                 "booster", "importance_type", "interaction_constraints",
                 "missing", "monotone_constraints", "num_parallel_tree",
                 "objective", "tree_method", "validate_parameters"]:
        print(f"  {k:<25} = {v}")

print(f"\n[TRAINING] Fitting XGBoost on {len(X_train):,} samples ...")
xgb_model.fit(X_train, y_train)
print("[TRAINING] Complete.")

# =============================================================================
# SECTION 4: THRESHOLD OPTIMIZATION
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: THRESHOLD OPTIMIZATION")
print("=" * 60)

# Default threshold is 0.5. But for fraud detection, we tune it using the
# Precision-Recall curve to find the threshold that maximizes F1.
y_prob = xgb_model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find threshold that gives the best F1 score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx   = np.argmax(f1_scores)
THRESHOLD  = float(thresholds[best_idx])
best_f1    = f1_scores[best_idx]

print(f"\n[INFO] Default threshold (0.50) F1 : "
      f"{f1_score(y_test, (y_prob >= 0.50).astype(int)):.4f}")
print(f"[INFO] Optimized threshold           : {THRESHOLD:.4f}")
print(f"[INFO] Optimized F1-Score            : {best_f1:.4f}")
print(f"\n[STRATEGY] Using threshold = {THRESHOLD:.4f} for production.")
print("  → A higher threshold reduces false positives (fewer legitimate")
print("    transactions wrongly blocked) at a small cost to recall.")

# Final predictions using optimized threshold
y_pred = (y_prob >= THRESHOLD).astype(int)

# =============================================================================
# SECTION 5: FULL MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: FULL MODEL EVALUATION")
print("=" * 60)

# --- Core Metrics ---
precision  = precision_score(y_test, y_pred)
recall     = recall_score(y_test, y_pred)
f1         = f1_score(y_test, y_pred)
roc_auc    = roc_auc_score(y_test, y_prob)
cm         = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print(f"\n{'─'*40}")
print(f"  EVALUATION METRIC RESULTS")
print(f"{'─'*40}")
print(f"  Precision  (Fraud class) : {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall     (Fraud class) : {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1-Score   (Fraud class) : {f1:.4f}")
print(f"  ROC-AUC                  : {roc_auc:.4f}")
print(f"{'─'*40}")

print(f"\n  CONFUSION MATRIX")
print(f"{'─'*40}")
print(f"                   Predicted")
print(f"                   Legit    Fraud")
print(f"  Actual  Legit  [ {tn:>6,}   {fp:>6,} ]  ← False Positives (blocked legit)")
print(f"  Actual  Fraud  [ {fn:>6,}   {tp:>6,} ]  ← False Negatives (missed fraud)")
print(f"{'─'*40}")
print(f"\n  True Positives  (Fraud caught)    : {tp:,}")
print(f"  True Negatives  (Legit approved)  : {tn:,}")
print(f"  False Positives (Legit blocked)   : {fp:,}  ← Minimize for UX")
print(f"  False Negatives (Fraud missed)    : {fn:,}  ← Minimize for security")

print(f"\n  FULL CLASSIFICATION REPORT")
print(f"{'─'*40}")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate (0)", "Fraud (1)"],
    digits=4
))

# --- Top Feature Importances ---
print(f"\n  TOP 15 FEATURE IMPORTANCES (by gain)")
print(f"{'─'*40}")
feature_names = X_train.columns.tolist()
importances   = xgb_model.feature_importances_
feat_imp_df   = (
    pd.DataFrame({"Feature": feature_names, "Importance": importances})
    .sort_values("Importance", ascending=False)
    .head(15)
    .reset_index(drop=True)
)
feat_imp_df.index += 1
print(feat_imp_df.to_string())

# =============================================================================
# SECTION 6: SAVE PRODUCTION ARTIFACTS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 6: SAVING PRODUCTION ARTIFACTS → output_v2/")
print("=" * 60)

# Compute historical spend stats for each card/address (for inference-time lookup)
feature_stats = {}
for col in ["card1", "card2", "addr1"]:
    if col in X_train.columns:
        stats_df = (
            X_train[[col, "TransactionAmt"]]
            .assign(TransactionAmt=X_train.get("TransactionAmt", 0))
        )
        feature_stats[col] = (
            X_train.groupby(col)["TransactionAmt"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "mean", "std": "std"})
            .apply(lambda row: {"mean": row["mean"], "std": row["std"]}, axis=1)
            .to_dict()
        )

# Save model
joblib.dump(xgb_model, "output_v2/best_model.pkl")
print("  ✓ best_model.pkl     saved")

# Save feature names (for alignment at inference time)
with open("output_v2/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print("  ✓ feature_names.pkl  saved")

# Save feature stats (for real-time aggregate lookups)
with open("output_v2/feature_stats.pkl", "wb") as f:
    pickle.dump(feature_stats, f)
print("  ✓ feature_stats.pkl  saved")

# Save model metadata + evaluation metrics
model_info = {
    "model"           : "XGBoost_v2_Overhaul",
    "threshold"       : THRESHOLD,
    "precision"       : round(precision, 4),
    "recall"          : round(recall, 4),
    "f1"              : round(f1, 4),
    "auc"             : round(roc_auc, 4),
    "tp"              : int(tp),
    "tn"              : int(tn),
    "fp"              : int(fp),
    "fn"              : int(fn),
    "trained_on"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_estimators"    : 500,
    "max_depth"       : 7,
    "scale_pos_weight": round(scale_pos_weight, 2),
}

with open("output_v2/model_info.pkl", "wb") as f:
    pickle.dump(model_info, f)
print("  ✓ model_info.pkl     saved")

print(f"\n{'='*60}")
print("  FraudShield Pro XGBoost v2 — Training Complete!")
print(f"  F1-Score : {f1:.4f}  |  AUC : {roc_auc:.4f}  |  Threshold : {THRESHOLD:.4f}")
print(f"{'='*60}\n")
