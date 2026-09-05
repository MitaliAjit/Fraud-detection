"""
app.py
-------
Flask API that serves the trained fraud detection model.

Run:
    python app.py
Then POST a transaction to /predict, e.g.:

    curl -X POST http://localhost:5000/predict \\
      -H "Content-Type: application/json" \\
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
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "models/fraud_model.joblib"
_model_bundle = None  # lazy-loaded


def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    bundle = get_model_bundle()
    pipeline = bundle["pipeline"]
    features = bundle["features"]

    payload = request.get_json(force=True)
    missing = [f for f in features if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    row = pd.DataFrame([{f: payload[f] for f in features}])
    fraud_probability = float(pipeline.predict_proba(row)[0, 1])
    is_fraud = bool(pipeline.predict(row)[0])

    # Simple risk banding on top of the raw probability
    if fraud_probability >= 0.75:
        risk_level = "high"
    elif fraud_probability >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    return jsonify({
        "is_fraud": is_fraud,
        "fraud_probability": round(fraud_probability, 4),
        "risk_level": risk_level,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
