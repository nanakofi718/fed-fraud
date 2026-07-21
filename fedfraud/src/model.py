"""
model.py
--------
The single model architecture shared by every experiment (centralized
baseline, local-only baselines, and federated clients), so comparisons
in Chapter Four are apples-to-apples.
"""
import torch
import torch.nn as nn


class FraudDetectionModel(nn.Module):
    """
    Simple feed-forward binary classifier.
    Note: no BatchNorm layers -- Opacus/DP-SGD requires per-sample
    gradients, which BatchNorm breaks (its statistics mix samples in a
    batch). GroupNorm/LayerNorm would be DP-compatible if you want
    normalization later; we use Dropout instead, which is DP-safe.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)  # returns logits (no sigmoid -- use BCEWithLogitsLoss)


def new_model(input_dim, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return FraudDetectionModel(input_dim)
