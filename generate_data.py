"""
generate_data.py
-----------------
Generates a synthetic transaction dataset for fraud detection.

Since public fraud datasets (e.g. Kaggle's credit card dataset) are large and
already PCA-anonymized, this script creates a realistic synthetic dataset
with interpretable features, so you can inspect, explain, and extend it.

Run:
    python generate_data.py
Output:
    data/transactions.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 20000
FRAUD_RATIO = 0.03  # ~3% fraud, mimics real-world class imbalance


def generate_dataset(n_samples: int = N_SAMPLES, fraud_ratio: float = FRAUD_RATIO,
                      seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def make_transactions(n, is_fraud: bool):
        if is_fraud:
            amount = rng.gamma(shape=2.0, scale=250, size=n)          # fraud skews higher
            hour = rng.choice(range(24), size=n, p=_night_heavy_hours())
            distance_from_home = rng.gamma(shape=3.0, scale=80, size=n)  # far from home
            time_since_last_txn = rng.exponential(scale=5, size=n)       # rapid-fire txns
            num_txns_last_hour = rng.poisson(lam=4, size=n)
            merchant_risk_score = rng.beta(a=5, b=2, size=n)              # risky merchants
            is_foreign = rng.choice([0, 1], size=n, p=[0.4, 0.6])
            card_present = rng.choice([0, 1], size=n, p=[0.85, 0.15])
        else:
            amount = rng.gamma(shape=2.0, scale=40, size=n)
            hour = rng.choice(range(24), size=n, p=_day_heavy_hours())
            distance_from_home = rng.gamma(shape=1.5, scale=5, size=n)
            time_since_last_txn = rng.exponential(scale=120, size=n)
            num_txns_last_hour = rng.poisson(lam=0.3, size=n)
            merchant_risk_score = rng.beta(a=2, b=5, size=n)
            is_foreign = rng.choice([0, 1], size=n, p=[0.95, 0.05])
            card_present = rng.choice([0, 1], size=n, p=[0.2, 0.8])

        return pd.DataFrame({
            "amount": amount.round(2),
            "hour_of_day": hour,
            "distance_from_home_km": distance_from_home.round(2),
            "time_since_last_txn_min": time_since_last_txn.round(2),
            "num_txns_last_hour": num_txns_last_hour,
            "merchant_risk_score": merchant_risk_score.round(3),
            "is_foreign_transaction": is_foreign,
            "card_present": card_present,
            "is_fraud": int(is_fraud),
        })

    df = pd.concat([
        make_transactions(n_legit, is_fraud=False),
        make_transactions(n_fraud, is_fraud=True),
    ], ignore_index=True)

    df = _add_label_noise(df, rng, flip_fraction=0.04)  # blur the boundary, more realistic
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    df.insert(0, "transaction_id", [f"TXN{100000 + i}" for i in range(len(df))])
    return df


def _add_label_noise(df: pd.DataFrame, rng: np.random.Generator, flip_fraction: float) -> pd.DataFrame:
    """
    Nudges a small fraction of each class toward the other class's feature
    distribution, simulating real-world ambiguity (e.g. a legitimate but
    unusual purchase, or a fraud that looks routine). Without this, the two
    classes are perfectly separable, which never happens with real data.
    """
    df = df.copy()
    n_flip_legit = int(flip_fraction * (df["is_fraud"] == 0).sum())
    n_flip_fraud = int(flip_fraction * (df["is_fraud"] == 1).sum())

    legit_idx = df[df["is_fraud"] == 0].sample(n=n_flip_legit, random_state=0).index
    fraud_idx = df[df["is_fraud"] == 1].sample(n=n_flip_fraud, random_state=1).index

    # Legit transactions that "look" a bit like fraud (e.g. a big trip purchase)
    df.loc[legit_idx, "amount"] *= rng.uniform(2.0, 4.0, size=len(legit_idx))
    df.loc[legit_idx, "distance_from_home_km"] *= rng.uniform(5.0, 15.0, size=len(legit_idx))
    df.loc[legit_idx, "merchant_risk_score"] = rng.uniform(0.5, 0.8, size=len(legit_idx))

    # Fraud transactions that "look" routine (e.g. a low-value test charge)
    df.loc[fraud_idx, "amount"] *= rng.uniform(0.1, 0.3, size=len(fraud_idx))
    df.loc[fraud_idx, "distance_from_home_km"] *= rng.uniform(0.05, 0.2, size=len(fraud_idx))
    df.loc[fraud_idx, "merchant_risk_score"] = rng.uniform(0.2, 0.5, size=len(fraud_idx))

    return df


def _night_heavy_hours():
    """Probability distribution over 24 hours, weighted toward late night (fraud pattern)."""
    weights = np.array([3 if h in list(range(0, 6)) else 1 for h in range(24)], dtype=float)
    return weights / weights.sum()


def _day_heavy_hours():
    """Probability distribution over 24 hours, weighted toward normal daytime activity."""
    weights = np.array([3 if h in list(range(8, 21)) else 1 for h in range(24)], dtype=float)
    return weights / weights.sum()


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(df)} transactions -> data/transactions.csv")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")
