
"""
Cora numerical study — GCN vs GAT.

Run:  python experiment.py
Output: results.csv  (one row per epoch × seed × model)
        summary printed to stdout
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.transforms import NormalizeFeatures

from cora_setup import load_cora_dataset

# ── Dataset ───────────────────────────────────────────────────────────────────
# Loaded once at module level; all runs share the same splits.
dataset = load_cora_dataset(root="data/Planetoid", transform=NormalizeFeatures())
data = dataset[0]

NUM_FEATURES = dataset.num_features  # 1433 bag-of-words node features
NUM_CLASSES  = dataset.num_classes   # 7 paper topics


# ── Models ────────────────────────────────────────────────────────────────────

class GCN(torch.nn.Module):
    """Two-layer GCN baseline (Kipf & Welling, ICLR 2017)."""

    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        self.conv1 = GCNConv(NUM_FEATURES, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, NUM_CLASSES)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    """Two-layer GAT (Veličković et al., ICLR 2018).

    Layer 1: 8 heads × 8 features, concatenated  →  64-dim
    Layer 2: 1 head, no concat                   →  NUM_CLASSES-dim logits

    Fix vs. original notebook: conv2 had heads=8 with concat=True (default),
    which produced [N, NUM_CLASSES * 8] = [N, 56] instead of [N, 7].
    Setting heads=1, concat=False corrects the output shape.
    """

    def __init__(self, hidden_channels: int = 8, heads: int = 8):
        super().__init__()
        self.conv1 = GATConv(NUM_FEATURES, hidden_channels, heads=heads)
        self.conv2 = GATConv(heads * hidden_channels, NUM_CLASSES, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ── Hyperparameters per model ─────────────────────────────────────────────────
# Central place to change lr / wd / architecture without touching training code.
MODEL_CONFIGS: dict[str, dict] = {
    "GCN": {
        "build": lambda: GCN(hidden_channels=16),
        "lr": 0.01,
        "weight_decay": 5e-4,
    },
    "GAT": {
        "build": lambda: GAT(hidden_channels=8, heads=8),
        "lr": 0.005,
        "weight_decay": 5e-4,
    },
}


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs so runs are fully reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Per-step helpers ──────────────────────────────────────────────────────────

def _train_step(model, optimizer, criterion) -> float:
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def _accuracy(model, mask) -> float:
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    return float((pred[mask] == data.y[mask]).sum()) / int(mask.sum())


# ── Single run ────────────────────────────────────────────────────────────────

def run_once(model_name: str, seed: int, epochs: int = 200) -> list[dict]:
    """Train one model for `epochs` epochs; return per-epoch metrics as rows."""
    set_seed(seed)
    cfg   = MODEL_CONFIGS[model_name]
    model = cfg["build"]()
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    crit  = torch.nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, epochs + 1):
        loss     = _train_step(model, opt, crit)
        val_acc  = _accuracy(model, data.val_mask)
        test_acc = _accuracy(model, data.test_mask)
        history.append({
            "model":    model_name,
            "seed":     seed,
            "epoch":    epoch,
            "loss":     loss,
            "val_acc":  val_acc,
            "test_acc": test_acc,
        })

    return history


# ── Multi-seed experiment ─────────────────────────────────────────────────────

def run_experiment(
    models: list[str] = list(MODEL_CONFIGS.keys()),
    seeds:  list[int] = [0, 1, 2, 3, 4],
    epochs: int       = 200,
    save_csv: str     = "results.csv",
) -> pd.DataFrame:
    """Run every model × seed combination and persist results to CSV."""
    rows = []
    for model_name in models:
        for seed in seeds:
            print(f"  {model_name}  seed={seed} ...", flush=True)
            rows.extend(run_once(model_name, seed, epochs))

    df = pd.DataFrame(rows)
    df.to_csv(save_csv, index=False)
    print(f"\nSaved {len(df)} rows → {save_csv}")
    return df


# ── Summary ───────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Report mean ± std test accuracy, evaluated at each seed's best val epoch."""
    # Pick the epoch with highest val_acc for each (model, seed) pair,
    # then read off its test_acc — avoids optimising directly on the test set.
    best = (
        df.sort_values("val_acc", ascending=False)
          .groupby(["model", "seed"], sort=False)
          .first()
          .reset_index()
    )
    return (
        best.groupby("model")["test_acc"]
            .agg(mean="mean", std="std", runs="count")
            .round(4)
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Cora numerical study ===")
    print(f"Nodes: {data.num_nodes}  |  Edges: {data.num_edges}  "
          f"|  Features: {NUM_FEATURES}  |  Classes: {NUM_CLASSES}\n")

    df = run_experiment()

    print("\nTest accuracy at best-val epoch (mean ± std, 5 seeds):")
    print(summarize(df).to_string())
