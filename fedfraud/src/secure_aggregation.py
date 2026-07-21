"""
secure_aggregation.py
----------------------
Implements a simplified version of the pairwise-masking secure
aggregation protocol (Bonawitz et al. 2017, cited in your Ch2 Section
2.5.4). The goal: the central server should only ever see the SUM of
client updates, never any individual client's update.

How it works:
  1. Every ordered pair of clients (i, j) agrees on a shared random
     seed (in a real deployment this comes from a Diffie-Hellman key
     exchange; here we simulate that agreement directly, since the
     cryptographic handshake itself is out of scope for this thesis --
     see your Scope section, which excludes Homomorphic Encryption/ZKPs
     as full cryptographic implementations).
  2. Client i adds a pseudorandom mask derived from seed(i,j) to its
     update for every other client j, and subtracts the mask derived
     from seed(j,i) for every j. Because seed(i,j) == seed(j,i), when
     the server sums ALL masked updates, every pairwise mask cancels
     out exactly, leaving only the true sum of updates.
  3. The server therefore recovers sum(updates) -- and from that the
     FedAvg weighted average -- without ever observing any individual
     client's raw update.

Limitation (state clearly in your Chapter Three/Five): this simplified
version does NOT implement Shamir secret-sharing for dropout resilience
(i.e. it assumes no client drops out mid-round). Bonawitz et al.'s full
protocol adds that; it's noted here as a direction for future work.
"""
import numpy as np
import torch


def _mask_seed(client_i, client_j, round_idx, master_seed=1234):
    """Deterministically derive a shared seed for pair (i, j) in a given round."""
    a, b = sorted((client_i, client_j))
    return master_seed + round_idx * 100000 + a * 1000 + b


def _pseudorandom_vector(seed, shape):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)


def apply_masks(client_updates, round_idx, master_seed=1234):
    """
    client_updates: list of state_dicts (one per participating client)
    Returns: list of MASKED state_dicts, same structure, ready to be summed
             by an untrusted/honest-but-curious server.
    """
    n = len(client_updates)
    masked_updates = [copy_state_dict(u) for u in client_updates]

    for key in client_updates[0].keys():
        shape = tuple(client_updates[0][key].shape)
        for i in range(n):
            mask_total = np.zeros(shape, dtype=np.float32)
            for j in range(n):
                if i == j:
                    continue
                seed = _mask_seed(i, j, round_idx, master_seed)
                mask = _pseudorandom_vector(seed, shape)
                mask_total += mask if i < j else -mask
            masked_updates[i][key] = client_updates[i][key] + torch.tensor(mask_total)
    return masked_updates


def copy_state_dict(sd):
    return {k: v.clone() for k, v in sd.items()}


def secure_sum(masked_updates):
    """
    What the server actually computes: a plain sum of masked updates.
    Because pairwise masks cancel across all clients, this equals the
    true sum of unmasked updates -- the server never sees any single
    client's real update at any point.
    """
    result = copy_state_dict(masked_updates[0])
    for key in result:
        for u in masked_updates[1:]:
            result[key] = result[key] + u[key]
    return result
