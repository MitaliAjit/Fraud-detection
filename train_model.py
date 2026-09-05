"""
train_model.py
----------------
Trains a fraud detection classifier on data/transactions.csv and saves the
trained model + evaluation report.

Handles class imbalance with class_weight='balanced' and reports metrics
that actually matter for fraud (precision, recall, ROC-AUC, PR-AUC) rather
than plain accuracy, which is misleading on imbalanced data.

Run:
    python train_model.py
Output:
    models/fraud_model.joblib
    models/metrics.json
    models/confusion_matrix.png
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)

FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "distance_from_home_km",
    "time_since_last_txn_min",
    "num_txns_last_hour",
    "merchant_risk_score",
    "is_foreign_transaction",
    "card_present",
]
TARGET_COLUMN = "is_fraud"
RANDOM_SEED = 42


def load_data(path: str = "data/transactions.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",   # key for imbalanced fraud data
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])


def main():
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["legit", "fraud"], output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_fraud": round(report["fraud"]["precision"], 4),
        "recall_fraud": round(report["fraud"]["recall"], 4),
        "f1_fraud": round(report["fraud"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
        "n_test_samples": int(len(y_test)),
        "n_fraud_in_test": int(y_test.sum()),
    }

    print("=== Evaluation on held-out test set ===")
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["legit", "fraud"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Fraud Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    plt.close()

    # Save model + metrics + feature order (needed at inference time)
    joblib.dump({"pipeline": pipeline, "features": FEATURE_COLUMNS}, "models/fraud_model.joblib")
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model      -> models/fraud_model.joblib")
    print("Saved metrics    -> models/metrics.json")
    print("Saved confusion  -> models/confusion_matrix.png")


if __name__ == "__main__":
    main()
