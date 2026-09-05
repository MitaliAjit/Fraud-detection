# 🛡️ Fraud Detection — ML Classification Project

A machine learning project that flags fraudulent transactions using a
Random Forest classifier, trained on a synthetic (but realistically noisy)
transaction dataset. Includes a Flask API to serve real-time predictions.

## Why this project

Fraud detection is a classic **imbalanced classification** problem — fraud
is rare (~3% of transactions here, similar to real-world rates), so plain
accuracy is meaningless (predicting "not fraud" every time would already
score ~97%). This project instead evaluates using **precision, recall,
ROC-AUC, and PR-AUC**, and handles imbalance with `class_weight="balanced"`.

## Features used

| Feature | Description |
|---|---|
| `amount` | Transaction amount |
| `hour_of_day` | Hour transaction occurred (0–23) |
| `distance_from_home_km` | Distance from cardholder's home location |
| `time_since_last_txn_min` | Minutes since the account's previous transaction |
| `num_txns_last_hour` | Transaction velocity — rapid-fire charges are a fraud signal |
| `merchant_risk_score` | Risk score of the merchant (0–1) |
| `is_foreign_transaction` | Whether the transaction is cross-border |
| `card_present` | Whether the physical card was used (in-person vs online) |

## Project structure

```
fraud-detection/
├── generate_data.py     # Creates a synthetic, realistically-noisy dataset
├── train_model.py       # Trains + evaluates the RandomForest classifier
├── app.py                # Flask API to serve predictions
├── requirements.txt
├── data/
│   └── transactions.csv  # Generated dataset (created by generate_data.py)
└── models/
    ├── fraud_model.joblib     # Trained model (created by train_model.py)
    ├── metrics.json           # Evaluation metrics on held-out test set
    └── confusion_matrix.png   # Visual evaluation
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Generate the dataset**
```bash
python generate_data.py
```

**2. Train the model**
```bash
python train_model.py
```
This prints evaluation metrics and saves the trained model + a confusion
matrix plot to `models/`.

**3. Serve predictions via API**
```bash
python app.py
```
Then send a transaction to check:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "amount": 899.50,
        "hour_of_day": 3,
        "distance_from_home_km": 220.0,
        "time_since_last_txn_min": 1.2,
        "num_txns_last_hour": 5,
        "merchant_risk_score": 0.81,
        "is_foreign_transaction": 1,
        "card_present": 0
      }'
```
Response:
```json
{
  "is_fraud": true,
  "fraud_probability": 0.97,
  "risk_level": "high"
}
```

## Model performance

On a held-out test set (20% of data, stratified):

| Metric | Score |
|---|---|
| ROC-AUC | ~0.999 |
| PR-AUC | ~0.997 |
| Precision (fraud) | ~0.98 |
| Recall (fraud) | ~0.98 |

*(exact numbers vary slightly by run — see `models/metrics.json` after training)*

## Notes on the dataset

The dataset is **synthetically generated** with intentional noise: ~4% of
legit transactions are nudged to look fraud-like (e.g. a big out-of-town
purchase) and vice versa, so the classes aren't perfectly separable — this
mirrors the ambiguity real fraud systems have to deal with. Swap in a real
dataset (e.g. Kaggle's Credit Card Fraud dataset) by matching the column
names in `FEATURE_COLUMNS` inside `train_model.py`.

## Possible extensions

- Swap Random Forest for XGBoost/LightGBM and compare
- Add SHAP explainability so each prediction shows *why* it was flagged
- Add a threshold-tuning script (precision/recall tradeoff for your use case)
- Containerize `app.py` with Docker for deployment
