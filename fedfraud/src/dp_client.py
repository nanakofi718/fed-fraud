"""
dp_client.py
------------
This replaces the "manual DP" from your draft, which clipped whole
parameter tensors after backward() -- that is NOT valid DP-SGD and
carries no formal privacy guarantee.

Real DP-SGD (Abadi et al. 2016, cited in your Ch2 Section 2.5.1) requires:
  1. PER-SAMPLE gradient clipping (each training example's gradient is
     clipped individually, BEFORE averaging into a batch gradient) --
     this bounds any single transaction's influence on the update.
  2. Calibrated Gaussian noise added to the CLIPPED, SUMMED gradient.
  3. A privacy accountant that tracks cumulative privacy loss (epsilon)
     across every local step, so you can report a real, defensible
     "(epsilon, delta)-DP" number in Chapter Four -- this is what
     Objective 3 ("formal privacy guarantees") actually requires.

Opacus (from Meta AI) implements exactly this and is the standard tool
for it in PyTorch, which is why it's in your recommended stack.
"""
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

from .model import new_model


class DPFederatedClient:
    def __init__(self, client_id, X, y, batch_size=64):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.num_samples = len(X)
        self.input_dim = X.shape[1]
        self.batch_size = min(batch_size, max(2, self.num_samples))

    def _make_loader(self):
        ds = TensorDataset(torch.FloatTensor(self.X), torch.FloatTensor(self.y))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    def train_local(self, global_state_dict, epochs, lr, max_grad_norm,
                     noise_multiplier, target_delta=1e-5, use_dp=True):
        """
        Trains locally starting from the current global weights.
        Returns: (state_dict, num_samples, epsilon_spent_this_round)

        use_dp=False lets you run the exact same code path WITHOUT
        privacy, useful as a "FL without DP" ablation to isolate how
        much accuracy DP costs you (this is your privacy-utility
        trade-off experiment, Objective 5).
        """
        model = new_model(self.input_dim)
        model.load_state_dict(copy.deepcopy(global_state_dict))
        model.train()

        loader = self._make_loader()

        pos = max(self.y.sum(), 1)
        neg = max(len(self.y) - self.y.sum(), 1)
        pos_weight = torch.tensor([min(neg / pos, 50.0)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        epsilon_spent = None

        if use_dp:
            privacy_engine = PrivacyEngine(accountant="rdp")
            model, optimizer, loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )

        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        if use_dp:
            epsilon_spent = privacy_engine.get_epsilon(delta=target_delta)
            # Opacus wraps the model in a GradSampleModule; unwrap to get
            # a plain state_dict compatible with the global model / FedAvg.
            trained_state = model._module.state_dict()
        else:
            trained_state = model.state_dict()

        return trained_state, self.num_samples, epsilon_spent
