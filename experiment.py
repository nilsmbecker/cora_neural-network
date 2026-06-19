
"""
Cora numerical study — GCN vs GAT, with a K / F' sweep.

Research question:
    How do the number of attention heads (K) and hidden feature dims (F')
    affect node-classification accuracy on Cora, and where does extra
    capacity start to overfit?

Design:
    One-at-a-time (OAT) sweep around the Veličković baseline (K=8, F'=8):
        - vary K  in {1, 2, 4, 8, 16}  with F'=8 fixed
        - vary F' in {4, 8, 16, 32, 64} with K=8 fixed
    Both lines cross at the (8, 8) baseline. A GCN(16) is included as a
    reference point. Phase 2 (combine the best K with the best F') is left
    for a follow-up run.

Run:    python experiment.py
Output: results.csv  (one row per epoch × seed × config)
        summary printed to stdout
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter

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

    def __init__(self, hidden_channels: int = 16, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(NUM_FEATURES, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, NUM_CLASSES)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    """Two-layer GAT (Veličković et al., ICLR 2018).

    Layer 1: K heads × F' features, concatenated  →  (K·F')-dim
    Layer 2: 1 head, no concat                    →  NUM_CLASSES-dim logits

    Dropout p is applied to the input and between layers (as in the paper).
    """

    def __init__(self, hidden_channels: int = 8, heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.conv1 = GATConv(NUM_FEATURES, hidden_channels, heads=heads)
        self.conv2 = GATConv(heads * hidden_channels, NUM_CLASSES, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ── Config layer: WHAT to run ──────────────────────────────────────────────────
# A Config fully describes one experiment. The sweep is just a list of Configs,
# and every result row is tagged with these fields so plots are one groupby away.

@dataclass
class Config:
    name: str            # unique label, e.g. "GAT_K8_F8"  → groups results
    kind: str            # "GAT" or "GCN" → for GAT-vs-GCN comparison
    K: int               # attention heads (1 for GCN)
    Fp: int              # hidden dim per head
    lr: float
    weight_decay: float
    dropout: float

    @property
    def KxFp(self) -> int:
        return self.K * self.Fp


def make_gat_config(K: int, Fp: int, lr: float = 0.005,
                    wd: float = 5e-4, dropout: float = 0.6) -> Config:
    return Config(name=f"GAT_K{K}_F{Fp}", kind="GAT", K=K, Fp=Fp,
                  lr=lr, weight_decay=wd, dropout=dropout)


def make_gcn_config(hidden: int = 16, lr: float = 0.01,
                    wd: float = 5e-4, dropout: float = 0.5) -> Config:
    return Config(name=f"GCN_H{hidden}", kind="GCN", K=1, Fp=hidden,
                  lr=lr, weight_decay=wd, dropout=dropout)


# Sweep axes. Baseline (8, 8) is shared by both lines.
K_SWEEP     = [1, 2, 4, 8, 16]
FP_SWEEP    = [4, 8, 16, 32, 64]
BASELINE_K  = 8
BASELINE_FP = 8


def build_sweep() -> list[Config]:
    """OAT sweep: vary K at F'=8, vary F' at K=8, plus a GCN baseline.

    Keyed by name in a dict so the shared (8, 8) point is trained only once.
    """
    configs: dict[str, Config] = {}
    for K in K_SWEEP:                       # K-sweep, F' fixed
        c = make_gat_config(K=K, Fp=BASELINE_FP)
        configs[c.name] = c
    for Fp in FP_SWEEP:                     # F'-sweep, K fixed
        c = make_gat_config(K=BASELINE_K, Fp=Fp)
        configs[c.name] = c
    gcn = make_gcn_config()
    configs[gcn.name] = gcn
    return list(configs.values())


# ── Model factory: Config → live model ─────────────────────────────────────────

def build_model(cfg: Config) -> torch.nn.Module:
    if cfg.kind == "GAT":
        return GAT(hidden_channels=cfg.Fp, heads=cfg.K, dropout=cfg.dropout)
    if cfg.kind == "GCN":
        return GCN(hidden_channels=cfg.Fp, dropout=cfg.dropout)
    raise ValueError(f"unknown kind: {cfg.kind!r}")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


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


# ── Single run: How to train one config ────────────────────────────────────────

def run_once(cfg: Config, seed: int, epochs: int = 200) -> list[dict]:
    """Train one config for `epochs` epochs; return per-epoch metric rows."""
    set_seed(seed)
    model    = build_model(cfg)
    opt      = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    crit     = torch.nn.CrossEntropyLoss()
    n_params = count_params(model)

    history = []
    for epoch in range(1, epochs + 1):
        t0   = perf_counter()
        loss = _train_step(model, opt, crit)
        dt   = perf_counter() - t0

        history.append({
            # identity / sweep tags
            "name":  cfg.name, "kind": cfg.kind,
            "K":     cfg.K,    "Fp":   cfg.Fp, "KxFp": cfg.KxFp,
            "seed":  seed,     "epoch": epoch,
            # learning metrics
            "loss":      loss,
            "train_acc": _accuracy(model, data.train_mask),
            "val_acc":   _accuracy(model, data.val_mask),
            "test_acc":  _accuracy(model, data.test_mask),
            # cost metrics
            "sec_per_epoch": dt,
            "params":        n_params,
        })

    return history


# ── Multi-config experiment ────────────────────────────────────────────────────

def run_experiment(
    configs: list[Config] | None = None,
    seeds:  list[int] = [0, 1, 2, 3, 4],
    epochs: int       = 200,
    save_csv: str     = "results.csv",
) -> pd.DataFrame:
    """Run every config × seed combination and persist results to CSV."""
    if configs is None:
        configs = build_sweep()

    rows = []
    for cfg in configs:
        for seed in seeds:
            print(f"  {cfg.name:14s}  seed={seed} ...", flush=True)
            rows.extend(run_once(cfg, seed, epochs))

    df = pd.DataFrame(rows)
    df.to_csv(save_csv, index=False)
    print(f"\nSaved {len(df)} rows → {save_csv}")
    return df


# ── Summary ───────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per config: mean ± std test acc at each seed's best-val epoch, plus the
    overfitting gap (train − val), convergence epoch, timing, and param count.
    """
    # Best-val epoch per (config, seed) — never selects on the test set.
    best = (
        df.sort_values("val_acc", ascending=False)
          .groupby(["name", "seed"], sort=False)
          .first()
          .reset_index()
    )
    best["gap"] = best["train_acc"] - best["val_acc"]  # overfitting proxy

    agg = best.groupby("name").agg(
        kind=("kind", "first"),
        K=("K", "first"),
        Fp=("Fp", "first"),
        KxFp=("KxFp", "first"),
        test_mean=("test_acc", "mean"),
        test_std=("test_acc", "std"),
        gap_mean=("gap", "mean"),
        conv_epoch=("epoch", "mean"),   # epoch of best val ≈ convergence
        params=("params", "first"),
        runs=("seed", "count"),
    )
    # Mean wall-clock per epoch over the *whole* run, not just the best epoch.
    agg["ms_per_epoch"] = df.groupby("name")["sec_per_epoch"].mean() * 1000

    return agg.round(4).sort_values(["kind", "KxFp"])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Cora numerical study (K / F' sweep) ===")
    print(f"Nodes: {data.num_nodes}  |  Edges: {data.num_edges}  "
          f"|  Features: {NUM_FEATURES}  |  Classes: {NUM_CLASSES}\n")

    df = run_experiment()

    print("\nTest accuracy at best-val epoch (mean ± std, 5 seeds):")
    print(summarize(df).to_string())
