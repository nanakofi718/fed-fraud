# Privacy-Preserving Collaborative Analytics for Financial Fraud Detection

Full implementation for Chapter Three (System Implementation) of your thesis.
Implements Federated Learning (FedAvg) + Gaussian Differential Privacy
(DP-SGD via Opacus) + Secure Aggregation, evaluated against centralized
and non-collaborative baselines, on PaySim / IEEE-CIS-style data.

## 1. Project layout

```
fedfraud/
├── data/                     <- put real paysim.csv / ieee_cis.csv here (optional)
├── src/
│   ├── data.py                Loading, synthetic generator, non-IID partitioning
│   ├── model.py                The shared neural network architecture
│   ├── metrics.py              AUC-ROC / AUC-PR / MCC evaluation
│   ├── baselines.py            Centralized + non-collaborative (local-only) models
│   ├── dp_client.py            Federated client with real per-sample DP-SGD (Opacus)
│   ├── secure_aggregation.py   Pairwise-masking secure aggregation protocol
│   └── federated.py            FedAvg orchestrator tying everything together
├── experiments/
│   └── run_experiment.py       Main script -- runs everything, saves results
└── results/                   <- CSV + plots land here after running
```

## 2. How the pieces fit together (read this first)

Your Chapter Three needs to demonstrate the full pipeline, not just one
model. Here's the story each file tells:

- **`data.py`** simulates the "multiple institutions" setup from your
  Scope. `partition_non_iid()` splits one dataset into N shards using a
  Dirichlet distribution — this is the standard way the FL literature
  creates realistic non-IID splits (some "banks" see almost no fraud,
  others see a lot), matching your Ch2 discussion of institutional
  heterogeneity.

- **`baselines.py`** gives you the two comparison points Objective 4
  requires: a **centralized** model (upper bound — what you'd get with
  no privacy constraint) and **local-only** models (lower bound — what
  each institution gets today, alone).

- **`dp_client.py`** is the fix for the flaw in your original code: it
  uses Opacus to do *real* DP-SGD — per-sample gradient clipping, not
  clipping whole parameter tensors after the fact — plus an actual
  **privacy accountant** that reports a defensible epsilon. This is what
  lets you write "(ε, δ)-DP guarantee" in your write-up and mean it.

- **`secure_aggregation.py`** implements the pairwise-masking idea from
  Bonawitz et al. (cited in your Ch2 §2.5.4): each client masks its
  update with random noise that exactly cancels when all updates are
  summed, so the server only ever sees the total, never any one
  institution's update.

- **`federated.py`** runs the communication rounds: send global weights →
  each client trains locally with DP-SGD → updates are masked → server
  sums the masked updates (secure aggregation) → server divides by
  total samples (FedAvg) → repeat. It also logs the simulated bytes
  transferred per round, for your communication-efficiency analysis
  (Objective 5).

- **`run_experiment.py`** runs all of the above in sequence and produces
  the table and plots you'll put directly into Chapter Four.

## 3. Quick start

```bash
cd fedfraud
pip install torch opacus scikit-learn pandas matplotlib

python3 experiments/run_experiment.py
```

That's it — it will auto-generate a synthetic PaySim-style dataset (since
no real file is present), run all four systems, and write results to
`results/`.

## 4. Using your REAL datasets

Download from Kaggle:
- **PaySim**: "Synthetic Financial Datasets For Fraud Detection"
- **IEEE-CIS**: the IEEE-CIS Fraud Detection competition data

Place the files at:
```
fedfraud/data/paysim.csv
fedfraud/data/ieee_cis.csv     (merge train_transaction + train_identity first if needed)
```

`data.py` automatically detects and uses the real file instead of the
synthetic generator — no code changes needed. Then run:

```bash
python3 experiments/run_experiment.py --dataset paysim
python3 experiments/run_experiment.py --dataset ieee_cis
```

## 5. Key command-line options (for your experiments section)

```bash
python3 experiments/run_experiment.py \
    --dataset paysim \
    --n_clients 5 \          # number of simulated institutions
    --alpha 1.0 \            # non-IID skew: lower = more skewed (try 0.1 for extreme)
    --rounds 15 \            # FL communication rounds
    --local_epochs 1 \       # local epochs per client per round
    --lr 0.05 \              # local learning rate
    --max_grad_norm 1.0 \    # DP-SGD clipping bound (C)
    --noise_levels 0.4 0.8 1.2 2.0   # sweep for the privacy-utility trade-off plot
```

Things worth varying for your Chapter Four experiments (each maps to an
objective):

| What to vary | Which objective it supports |
|---|---|
| `--noise_levels` (wider range, e.g. `0.2 0.5 1.0 1.5 2.0 3.0`) | Objective 5: privacy-utility trade-off |
| `--n_clients` (e.g. 3, 5, 10, 20) | Objective 5: communication efficiency vs. # institutions |
| `--alpha` (e.g. 0.1, 0.5, 1.0, 10) | Objective 2/4: robustness to non-IID data |
| `--max_grad_norm` (e.g. 0.5, 1.0, 2.0, 5.0) | Discussion: clipping bound's own effect on utility |

Run several configurations, save each `results/summary_table.csv` under a
different name (e.g. `results_alpha0.1.csv`), and you'll have exactly the
comparative tables Chapter Four needs.

## 6. Outputs you get after running

- `results/summary_table.csv` — every system's AUC-ROC / AUC-PR / MCC /
  epsilon / communication cost in one table.
- `results/fl_vs_baselines.png` — bar chart comparing all systems.
- `results/privacy_utility_tradeoff.png` — AUC-ROC vs. epsilon, with the
  centralized and local-only bounds drawn as reference lines. This is
  probably your single most important thesis figure.
- `results/communication_cost.png` — simulated MB transferred per system.

## 7. Reading the epsilon numbers correctly

- Lower epsilon = **stronger** privacy (less information can leak about
  any single transaction).
- Typical published "reasonable" ranges for DP-SGD are roughly ε ≈ 1–10
  for meaningful protection; ε > 20 offers only weak guarantees in
  practice even though it's technically "differentially private."
- If your results show a steep AUC drop even at large epsilon, that's
  usually the **clipping bound** (`max_grad_norm`), not the noise, doing
  the damage — worth explicitly separating in your discussion (you can
  test this by running with `noise_multiplier=0` in code but leaving
  clipping on, isolating clipping's effect from noise's effect).

## 8. Known simplification to disclose in your Limitations section

`secure_aggregation.py` implements pairwise masking **without** Shamir
secret-sharing dropout recovery, i.e. it assumes no client drops out
mid-round. The full Bonawitz et al. protocol handles dropout; this is a
reasonable and common simplification for a thesis-level simulation, but
should be stated explicitly as a scope boundary, not silently omitted.

## 9. Extending this later (optional, not required for the thesis)

- Swap the hand-rolled FedAvg loop for the **Flower** framework
  (`flwr`) if you want a more "production" story in Chapter Three —
  the model, client, and DP logic here would carry over largely unchanged.
- Add FedProx (proximal term) as a second aggregation strategy to
  directly test the non-IID convergence issue your Ch2 literature
  review raises.
