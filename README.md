# Fraud Detection - ML Classification Project

A machine learning project that flags fraudulent transactions using a Random Forest classifier, trained on a synthetic but realistically noisy transaction dataset. Includes a Flask API to serve real-time predictions.

## Why this project

Fraud detection is a classic **imbalanced classification** problem - fraud is rare (about 3% of transactions here), so plain accuracy is not enough.

This project evaluates the model using **precision, recall, ROC-AUC, and PR-AUC**, and handles class imbalance with `class_weight="balanced"`.

## Features used

| Feature | Description |
|---|---|
| `amount` | Transaction amount |
| `hour_of_day` | Hour transaction occurred (0-23) |
| `distance_from_home_km` | Distance from cardholder's home location |
| `time_since_last_txn_min` | Minutes since the account's previous transaction |
| `num_txns_last_hour` | Transaction velocity |
| `merchant_risk_score` | Risk score of the merchant (0-1) |
| `is_foreign_transaction` | Whether the transaction is cross-border |
| `card_present` | Whether the physical card was used |

## Project structure

```text
fraud-detection/
|-- generate_data.py
|-- train_model.py
|-- app.py
|-- requirements.txt
|-- data/
|   `-- transactions.csv
`-- models/
    |-- fraud_model.joblib
    |-- metrics.json
    `-- confusion_matrix.png