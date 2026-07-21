"""
run_experiment.py
------------------
This is the main entry point. It runs everything your Chapter Three/Four
needs and saves results to ../results/:

  1. Centralized baseline           (Objective 4 upper bound)
  2. Non-collaborative local models (Objective 4 lower bound)
  3. Federated Learning, no DP      (isolates the FL communication gain)
  4. Federated Learning + DP-SGD + Secure Aggregation, at several noise
     levels (Objective 5: privacy-utility trade-off)

Usage:
    cd fedfraud
    python3 experiments/run_experiment.py --dataset paysim --n_clients 5 --rounds 15

Outputs:
    results/summary_table.csv
    results/privacy_utility_tradeoff.png
    results/fl_vs_baselines.png
    results/communication_cost.png
"""
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data import load_paysim, load_ieee_cis, partition_non_iid
from src.baselines import run_centralized_baseline, run_local_only_baselines
from src.federated import run_federated_training
from src.metrics import format_metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main(args):
    print(f"\n=== Loading dataset: {args.dataset} ===")
    if args.dataset == "paysim":
        X, y, cols = load_paysim()
    elif args.dataset == "ieee_cis":
        X, y, cols = load_ieee_cis()
    else:
        raise ValueError("dataset must be 'paysim' or 'ieee_cis'")

    print(f"Total samples: {len(y)}, fraud rate: {y.mean():.4%}, features: {len(cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clients = partition_non_iid(X_train, y_train, n_clients=args.n_clients,
                                 alpha=args.alpha, seed=42)
    print(f"\nPartitioned into {args.n_clients} simulated institutions (alpha={args.alpha}):")
    for i, (cx, cy) in enumerate(clients):
        print(f"  institution {i}: n={len(cy)}, fraud_rate={cy.mean():.4f}")

    results_rows = []

    # ---------------------------------------------------------------
    # 1. Centralized baseline
    # ---------------------------------------------------------------
    print("\n=== [1/4] Centralized baseline ===")
    _, cent_metrics = run_centralized_baseline(X_train, y_train, X_test, y_test,
                                                epochs=args.local_epochs * 3)
    print(format_metrics("Centralized (upper bound)", cent_metrics))
    results_rows.append({"system": "Centralized", "epsilon": None,
                          "comm_mb": None, **cent_metrics})

    # ---------------------------------------------------------------
    # 2. Non-collaborative local-only baselines
    # ---------------------------------------------------------------
    print("\n=== [2/4] Non-collaborative (local-only) baselines ===")
    local_results, local_avg = run_local_only_baselines(clients, X_test, y_test,
                                                          epochs=args.local_epochs * 3)
    print(format_metrics("Local-only (avg, lower bound)", local_avg))
    results_rows.append({"system": "Local-only (avg)", "epsilon": None,
                          "comm_mb": None, **local_avg})

    # ---------------------------------------------------------------
    # 3. Federated Learning, no DP (communication-only gain)
    # ---------------------------------------------------------------
    print("\n=== [3/4] Federated Learning (FedAvg, no DP) ===")
    fl_out = run_federated_training(clients, X_test, y_test,
                                     comm_rounds=args.rounds,
                                     local_epochs=args.local_epochs,
                                     lr=args.lr, use_dp=False,
                                     use_secure_agg=True, verbose=True)
    print(format_metrics("FL, no DP", fl_out["final_metrics"]))
    results_rows.append({"system": "FL (no DP)", "epsilon": None,
                          "comm_mb": fl_out["total_comm_mb"], **fl_out["final_metrics"]})

    # ---------------------------------------------------------------
    # 4. Federated Learning + DP-SGD + Secure Aggregation
    #    at several noise levels -> privacy-utility trade-off
    # ---------------------------------------------------------------
    print("\n=== [4/4] Federated Learning + DP-SGD + Secure Aggregation ===")
    tradeoff_rows = []
    for noise in args.noise_levels:
        print(f"\n-- noise_multiplier = {noise} --")
        dp_out = run_federated_training(clients, X_test, y_test,
                                         comm_rounds=args.rounds,
                                         local_epochs=args.local_epochs,
                                         lr=args.lr,
                                         max_grad_norm=args.max_grad_norm,
                                         noise_multiplier=noise,
                                         use_dp=True, use_secure_agg=True,
                                         verbose=True)
        label = f"FL+DP (noise={noise})"
        print(format_metrics(label, dp_out["final_metrics"]))
        row = {"system": label, "epsilon": dp_out["final_epsilon"],
               "comm_mb": dp_out["total_comm_mb"], **dp_out["final_metrics"]}
        results_rows.append(row)
        tradeoff_rows.append(row)

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    df = pd.DataFrame(results_rows)
    csv_path = os.path.join(RESULTS_DIR, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary table -> {csv_path}")
    print(df[["system", "epsilon", "comm_mb", "auc_roc", "auc_pr", "mcc"]]
          .to_string(index=False))

    _plot_privacy_utility_tradeoff(tradeoff_rows, cent_metrics, local_avg)
    _plot_fl_vs_baselines(df)
    _plot_communication_cost(df)


def _plot_privacy_utility_tradeoff(tradeoff_rows, cent_metrics, local_avg):
    if not tradeoff_rows:
        return
    epsilons = [r["epsilon"] for r in tradeoff_rows]
    auc_rocs = [r["auc_roc"] for r in tradeoff_rows]

    plt.figure(figsize=(7, 5))
    order = np.argsort(epsilons)
    plt.plot(np.array(epsilons)[order], np.array(auc_rocs)[order],
              marker="o", label="FL + DP-SGD")
    plt.axhline(cent_metrics["auc_roc"], color="green", linestyle="--",
                label="Centralized (no privacy)")
    plt.axhline(local_avg["auc_roc"], color="red", linestyle="--",
                label="Local-only (no collaboration)")
    plt.xlabel("Privacy spend (epsilon) -- lower epsilon = stronger privacy")
    plt.ylabel("AUC-ROC on held-out test set")
    plt.title("Privacy-Utility Trade-off (Objective 5)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "privacy_utility_tradeoff.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot -> {out_path}")


def _plot_fl_vs_baselines(df):
    plt.figure(figsize=(8, 5))
    plt.bar(df["system"], df["auc_roc"])
    plt.ylabel("AUC-ROC")
    plt.title("System Comparison (Objective 4)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "fl_vs_baselines.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot -> {out_path}")


def _plot_communication_cost(df):
    sub = df.dropna(subset=["comm_mb"])
    if sub.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.bar(sub["system"], sub["comm_mb"])
    plt.ylabel("Total simulated communication (MB)")
    plt.title("Communication Cost (Objective 5)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "communication_cost.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="paysim", choices=["paysim", "ieee_cis"])
    parser.add_argument("--n_clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0,
                         help="Dirichlet non-IID skew (lower = more skewed)")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--noise_levels", type=float, nargs="+",
                         default=[0.4, 0.8, 1.2, 2.0])
    args = parser.parse_args()
    main(args)
