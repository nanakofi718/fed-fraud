"""
federated.py
------------
The central orchestrator: runs communication rounds of FedAvg where
each client trains with real DP-SGD (dp_client.py) and updates are
combined via secure aggregation (secure_aggregation.py) before the
server averages them. Also logs communication cost (Objective 5:
communication efficiency).
"""
import copy
import time
import numpy as np
import torch

from .model import new_model
from .metrics import evaluate_model
from .dp_client import DPFederatedClient
from .secure_aggregation import apply_masks, secure_sum, copy_state_dict


def _state_dict_size_mb(state_dict):
    n_params = sum(v.numel() for v in state_dict.values())
    return n_params * 4 / (1024 ** 2)  # float32 = 4 bytes


class FederatedServer:
    def __init__(self, input_dim, seed=42):
        self.global_model = new_model(input_dim, seed=seed)

    def get_weights(self):
        return copy.deepcopy(self.global_model.state_dict())

    def set_weights(self, state_dict):
        self.global_model.load_state_dict(state_dict)

    def fedavg_from_secure_sum(self, summed_state, sample_counts):
        """
        Given the SUM of (weighted) client updates recovered via secure
        aggregation, divide by total samples to get the FedAvg weighted
        average, and load it as the new global model.
        """
        total = sum(sample_counts)
        new_state = {k: v / total for k, v in summed_state.items()}
        self.set_weights(new_state)


def run_federated_training(clients_data, X_test, y_test, comm_rounds=15,
                            local_epochs=1, lr=0.05, max_grad_norm=1.0,
                            noise_multiplier=0.0, use_dp=True,
                            use_secure_agg=True, seed=42, verbose=True):
    """
    clients_data: list of (X_c, y_c) tuples, one per simulated institution
                  (output of data.partition_non_iid).
    noise_multiplier=0.0 with use_dp=True effectively disables noise but
        still exercises the DP-SGD per-sample-clipping code path -- set
        use_dp=False instead for a true no-DP ablation (faster, and
        avoids Opacus overhead entirely).

    Returns a dict with the trained model, round-by-round test metrics,
    total simulated communication cost, and the final privacy spend.
    """
    input_dim = clients_data[0][0].shape[1]
    server = FederatedServer(input_dim, seed=seed)
    clients = [DPFederatedClient(i, X, y) for i, (X, y) in enumerate(clients_data)]

    history = []
    total_bytes_mb = 0.0
    max_epsilon = 0.0
    t0 = time.time()

    for round_idx in range(comm_rounds):
        global_weights = server.get_weights()
        round_updates = []
        sample_counts = []

        for client in clients:
            state, n_samples, epsilon = client.train_local(
                global_state_dict=global_weights,
                epochs=local_epochs,
                lr=lr,
                max_grad_norm=max_grad_norm,
                noise_multiplier=noise_multiplier,
                use_dp=use_dp,
            )
            # Weight each client's update by its sample count BEFORE
            # secure aggregation, so a plain sum afterwards gives the
            # correct FedAvg weighted average.
            weighted_state = {k: v * n_samples for k, v in state.items()}
            round_updates.append(weighted_state)
            sample_counts.append(n_samples)
            total_bytes_mb += _state_dict_size_mb(state) * 2  # download + upload
            if epsilon is not None:
                max_epsilon = max(max_epsilon, epsilon)

        if use_secure_agg:
            masked = apply_masks(round_updates, round_idx=round_idx)
            summed = secure_sum(masked)
        else:
            summed = copy_state_dict(round_updates[0])
            for key in summed:
                for u in round_updates[1:]:
                    summed[key] = summed[key] + u[key]

        server.fedavg_from_secure_sum(summed, sample_counts)

        metrics = evaluate_model(server.global_model, X_test, y_test)
        metrics["round"] = round_idx + 1
        metrics["epsilon_so_far"] = max_epsilon if use_dp else None
        history.append(metrics)

        if verbose:
            eps_str = f"eps={max_epsilon:.2f}" if use_dp else "eps=inf (no DP)"
            print(f"  round {round_idx+1:>2}/{comm_rounds} | "
                  f"AUC-ROC={metrics['auc_roc']:.4f} AUC-PR={metrics['auc_pr']:.4f} "
                  f"| {eps_str}")

    elapsed = time.time() - t0
    return {
        "model": server.global_model,
        "history": history,
        "final_metrics": history[-1],
        "total_comm_mb": total_bytes_mb,
        "final_epsilon": max_epsilon if use_dp else None,
        "wall_clock_sec": elapsed,
        "n_clients": len(clients),
        "comm_rounds": comm_rounds,
    }
