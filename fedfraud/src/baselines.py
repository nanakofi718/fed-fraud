"""
baselines.py
------------
Objective 4 of your thesis requires comparing the FL system against:

  (a) a CENTRALIZED model  -- trained on all institutions' pooled data.
      This is the "upper bound": what you'd get if privacy/regulation
      were not a constraint. FL + DP should approach this, not beat it.

  (b) NON-COLLABORATIVE (local-only) models -- one model trained per
      institution, on only its own data, never sharing anything. This
      is the "lower bound" / status quo you're arguing FL improves on.
"""
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import new_model
from .metrics import evaluate_model


def _make_loader(X, y, batch_size=64, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_plain_model(X_train, y_train, input_dim, epochs=10, lr=0.001,
                       batch_size=64, seed=42, class_weight=True):
    """
    Standard (non-private) Adam training used by both baselines.

    Note: a naive pos_weight = (neg_count / pos_count) can be in the
    hundreds at a <1% fraud rate. Combined with a not-small learning
    rate and SGD+momentum, this reliably pushes every ReLU unit into
    its dead (always-negative) region on an early large update, after
    which the whole network collapses to a single constant output
    regardless of input -- the model looks "trained" (loss goes down,
    no NaNs) but has actually flatlined. We cap pos_weight and use
    Adam (adaptive per-parameter step size), which is far more robust
    to this failure mode.
    """
    model = new_model(input_dim, seed=seed)
    loader = _make_loader(X_train, y_train, batch_size=batch_size)

    if class_weight:
        pos = max(y_train.sum(), 1)
        neg = max(len(y_train) - y_train.sum(), 1)
        pos_weight = torch.tensor([min(neg / pos, 50.0)])
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model


def run_centralized_baseline(X_train, y_train, X_test, y_test, epochs=10, lr=0.01, seed=42):
    """Train one model on ALL pooled institution data (the upper bound)."""
    input_dim = X_train.shape[1]
    model = train_plain_model(X_train, y_train, input_dim, epochs=epochs, lr=lr, seed=seed)
    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics


def run_local_only_baselines(clients, X_test, y_test, epochs=10, lr=0.01, seed=42):
    """
    Train one model PER institution, using only that institution's data,
    then evaluate every local model on the same shared test set. Returns
    both per-institution metrics and the average (the non-collaborative
    lower bound referenced in Objective 4).
    """
    input_dim = clients[0][0].shape[1]
    results = []
    for i, (X_c, y_c) in enumerate(clients):
        if y_c.sum() == 0:
            print(f"[baseline-local] institution {i} has zero fraud examples "
                  f"in its shard -- skipping local training (this itself is a "
                  f"data-siloing finding worth reporting).")
            continue
        model = train_plain_model(X_c, y_c, input_dim, epochs=epochs, lr=lr, seed=seed + i)
        m = evaluate_model(model, X_test, y_test)
        m["institution"] = i
        m["n_train"] = len(y_c)
        results.append(m)

    avg = {
        "auc_roc": sum(r["auc_roc"] for r in results) / len(results),
        "auc_pr": sum(r["auc_pr"] for r in results) / len(results),
        "mcc": sum(r["mcc"] for r in results) / len(results),
        "n_test": len(y_test),
        "fraud_rate_test": float(y_test.mean()),
    }
    return results, avg
