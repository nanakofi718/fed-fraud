"""
metrics.py
----------
Your literature review (Section 2.7.3) commits you to AUC-ROC as the
primary metric and AUC-PR (+ MCC) as secondary metrics, because plain
accuracy is meaningless on a <1% fraud rate dataset. This module
computes all three consistently for every experiment.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef


@torch.no_grad()
def evaluate_model(model, X_test, y_test, device="cpu", threshold=0.5):
    model.eval()
    model.to(device)
    X_t = torch.FloatTensor(X_test).to(device)
    logits = model(X_t).squeeze(1)
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)

    y_true = np.asarray(y_test)
    # AUC metrics are undefined if the eval set has only one class
    if len(np.unique(y_true)) < 2:
        auc_roc, auc_pr, mcc = float("nan"), float("nan"), float("nan")
    else:
        auc_roc = roc_auc_score(y_true, probs)
        auc_pr = average_precision_score(y_true, probs)
        mcc = matthews_corrcoef(y_true, preds)

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "mcc": mcc,
        "fraud_rate_test": float(y_true.mean()),
        "n_test": len(y_true),
    }


def format_metrics(name, m):
    return (f"{name:<28} | AUC-ROC={m['auc_roc']:.4f}  AUC-PR={m['auc_pr']:.4f}  "
            f"MCC={m['mcc']:.4f}  (n={m['n_test']}, fraud_rate={m['fraud_rate_test']:.4f})")
