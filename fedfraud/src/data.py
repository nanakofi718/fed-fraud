"""
data.py
-------
Handles three things for the thesis system:

1. Loading REAL PaySim / IEEE-CIS data if you have downloaded the CSVs
   from Kaggle and placed them in the data/ folder.
2. Generating a SYNTHETIC PaySim-style dataset if no real file is found,
   so the whole pipeline is runnable end-to-end without Kaggle access.
3. Partitioning a dataset into non-IID "institution" shards, simulating
   multiple banks/mobile money operators (as described in your Scope,
   Section 2.4.4, and Section 2.7 of Chapter Two).

Expected real files (place in fedfraud/data/):
    - paysim.csv        (Kaggle: "Synthetic Financial Datasets For Fraud Detection")
      required columns: step, type, amount, nameOrig, oldbalanceOrg,
      newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud
    - ieee_cis.csv       (merged train_transaction + train_identity from the
      IEEE-CIS Kaggle competition, or just train_transaction.csv)
      required column: isFraud (target)

If these are not found, a synthetic PaySim-style dataset is generated
using the same field structure and a realistic ~0.4% fraud rate, so you
can develop and test before your real data is ready.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---------------------------------------------------------------------------
# 1. SYNTHETIC PAYSIM-STYLE GENERATOR
# ---------------------------------------------------------------------------
def generate_synthetic_paysim(n_rows=60000, fraud_rate=0.004, seed=42):
    """
    Generates a synthetic dataset that mimics PaySim's structure and
    statistical properties (mobile-money transfers, severe class
    imbalance, transaction-type-dependent fraud). This is a stand-in for
    the real PaySim.csv from Kaggle — swap it out by placing the real
    file at data/paysim.csv.
    """
    rng = np.random.default_rng(seed)
    types = np.array(["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"])
    type_probs = np.array([0.35, 0.34, 0.22, 0.08, 0.01])

    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    def make_rows(n, fraud):
        # Fraud in PaySim is concentrated in TRANSFER / CASH_OUT transactions
        if fraud:
            tx_type = rng.choice(["TRANSFER", "CASH_OUT"], size=n, p=[0.5, 0.5])
            amount = rng.lognormal(mean=12.5, sigma=1.2, size=n)  # larger amounts
        else:
            tx_type = rng.choice(types, size=n, p=type_probs)
            amount = rng.lognormal(mean=9.5, sigma=1.5, size=n)

        old_bal_org = rng.exponential(scale=50000, size=n)
        # Fraudulent transfers often empty the account
        new_bal_org = np.where(
            fraud, np.maximum(0, old_bal_org - amount) * rng.uniform(0, 0.05, n),
            np.maximum(0, old_bal_org - amount)
        )
        old_bal_dest = rng.exponential(scale=30000, size=n)
        new_bal_dest = old_bal_dest + amount * rng.uniform(0.8, 1.0, n)

        step = rng.integers(1, 744, size=n)  # 744 hourly steps ~= 31 days
        return pd.DataFrame({
            "step": step,
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": old_bal_org,
            "newbalanceOrig": new_bal_org,
            "oldbalanceDest": old_bal_dest,
            "newbalanceDest": new_bal_dest,
            "isFraud": int(fraud),
        })

    df = pd.concat([make_rows(n_legit, False), make_rows(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. LOADING (real file if present, else synthetic)
# ---------------------------------------------------------------------------
def load_paysim():
    real_path = os.path.join(DATA_DIR, "paysim.csv")
    if os.path.exists(real_path):
        print(f"[data] Loading REAL PaySim data from {real_path}")
        df = pd.read_csv(real_path)
    else:
        print("[data] Real paysim.csv not found -> generating synthetic PaySim-style data.")
        print(f"[data] To use the real dataset, place it at: {real_path}")
        df = generate_synthetic_paysim()
    return _engineer_paysim_features(df)


def load_ieee_cis():
    real_path = os.path.join(DATA_DIR, "ieee_cis.csv")
    if os.path.exists(real_path):
        print(f"[data] Loading REAL IEEE-CIS data from {real_path}")
        df = pd.read_csv(real_path)
        return _engineer_ieee_features(df)
    else:
        print("[data] Real ieee_cis.csv not found -> generating a synthetic "
              "IEEE-CIS-style dataset instead (same idea as PaySim generator, "
              "different fraud rate/shape).")
        print(f"[data] To use the real dataset, place it at: {real_path}")
        df = generate_synthetic_paysim(n_rows=50000, fraud_rate=0.035, seed=7)
        return _engineer_paysim_features(df)


def _engineer_paysim_features(df):
    """Turn raw PaySim columns into a numeric feature matrix + label."""
    df = df.copy()
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df = pd.get_dummies(df, columns=["type"], prefix="type")

    feature_cols = [c for c in df.columns if c not in
                    ("isFraud", "nameOrig", "nameDest", "isFlaggedFraud")]
    X = df[feature_cols].astype(float).fillna(0.0)
    y = df["isFraud"].astype(float)
    return X.values, y.values, list(X.columns)


def _engineer_ieee_features(df):
    """Minimal numeric-only feature engineering for the real IEEE-CIS file."""
    df = df.copy()
    y = df["isFraud"].astype(float)
    X = df.drop(columns=["isFraud"])
    # Keep numeric columns only; encode a handful of low-cardinality categoricals
    non_numeric = X.select_dtypes(include=["object"]).columns
    for c in non_numeric:
        if X[c].nunique() <= 15:
            X = pd.get_dummies(X, columns=[c], prefix=c)
        else:
            X = X.drop(columns=[c])
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    return X.values, y.values, list(X.columns)


# ---------------------------------------------------------------------------
# 3. NON-IID PARTITIONING ACROSS SIMULATED INSTITUTIONS
# ---------------------------------------------------------------------------
def partition_non_iid(X, y, n_clients=5, alpha=1.0, seed=42, min_size=50):
    """
    Splits (X, y) into n_clients non-IID shards using a Dirichlet
    distribution over the class label (standard non-IID partitioning
    method in the FL literature, e.g. Hsu et al. / your Ch2 discussion
    of non-IID data across financial institutions).

    alpha controls heterogeneity:
        alpha -> large (e.g. 100)  => shards look close to IID
        alpha -> small (e.g. 0.1)  => shards are highly skewed
                                       (some institutions see almost no fraud)
    """
    n_classes = 2
    for attempt in range(20):
        rng = np.random.default_rng(seed + attempt)
        idx_by_class = [np.where(y == c)[0] for c in range(n_classes)]
        for idx in idx_by_class:
            rng.shuffle(idx)

        client_indices = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            proportions = rng.dirichlet(alpha=np.repeat(alpha, n_clients))
            proportions = (np.cumsum(proportions) * len(idx_by_class[c])).astype(int)[:-1]
            splits = np.split(idx_by_class[c], proportions)
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_size:
            break
    else:
        print(f"[data] Warning: could not guarantee min_size={min_size} for all "
              f"clients after 20 attempts; proceeding with smallest sizes={sizes}")

    clients = []
    for indices in client_indices:
        indices = np.array(indices)
        rng.shuffle(indices)
        clients.append((X[indices], y[indices]))
    return clients


if __name__ == "__main__":
    X, y, cols = load_paysim()
    print("PaySim feature matrix:", X.shape, "fraud rate:", y.mean())
    clients = partition_non_iid(X, y, n_clients=5, alpha=0.5)
    for i, (cx, cy) in enumerate(clients):
        print(f"  institution {i}: n={len(cy)}, fraud_rate={cy.mean():.4f}")
