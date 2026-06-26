# Cora GAT Experiment — K / F' Sweep

**Course:** Digital Operations — Advanced Seminar, SS26
**Authors:** J. Lennart Hunsicker; Nils Becker

## Research Question

> How do the number of attention heads (K) and hidden feature dimensions (F') affect node-classification accuracy on the Cora dataset, and at what capacity threshold does overfitting begin?

## Setup

```bash
python -m venv .venv
.venv/bin/pip install torch-geometric certifi scipy scikit-learn pandas matplotlib ipywidgets

# Verify data loading
.venv/bin/python -c "from cora_setup import load_cora_dataset; d = load_cora_dataset(); print(d[0])"
```

## Running the Experiment

```bash
# Runs all configs × 5 seeds, saves to output/results.csv (~10 min)
.venv/bin/python experiment.py
```

Then open `main.ipynb` to visualize results.

## Experiment Design

One-at-a-time (OAT) sweep around the Veličković (2018) baseline (K=8, F'=8):

| Sweep | Fixed | Values |
|---|---|---|
| K (attention heads) | F'=8 | 1, 2, 4, 8, 16 |
| F' (hidden dims/head) | K=8 | 4, 8, 16, 32, 64 |

Plus a GCN baseline (hidden=16) as reference. Both sweep lines cross at the (8, 8) baseline, so 10 unique configs total. Each config runs over 5 seeds × 500 epochs.

**Metrics per config:** test/val/train accuracy & loss, overfitting gap (test_loss − train_loss), MAD (oversmoothing), FLOPs, parameter count, wall-clock time/epoch.

## Current Status

| Step | Status |
|---|---|
| experiment.py implemented (models, metrics, sweep) | Done |
| main.ipynb visualization cells | Done |
| Partial sweep run (5/10 configs) | Done |
| Full sweep (expand K_SWEEP / FP_SWEEP and re-run) | **TODO** |
| Analysis and write-up | TODO |

**Missing configs:** GAT_K4_F8, GAT_K16_F8, GAT_K8_F16, GAT_K8_F32, GAT_K8_F64

To run the full sweep, set in `experiment.py`:
```python
K_SWEEP  = [1, 2, 4, 8, 16]
FP_SWEEP = [4, 8, 16, 32, 64]
```
then re-run `python experiment.py`.

## File Overview

| File | Purpose |
|---|---|
| `experiment.py` | All training logic — models, sweep, metrics |
| `main.ipynb` | Visualization notebook (loads output/results.csv) |
| `cora_setup.py` | Downloads Cora raw files for PyG |
| `output/results.csv` | Results (one row per epoch × seed × config) |
| `output/*.png` | Figures: sweep curves, overfitting, cost, t-SNE |

## Baseline Reference

Veličković et al. (2018) — GAT on Cora: K=8, F'=8, dropout=0.6, Adam lr=0.005, weight_decay=5e-4, reported ~83% test accuracy.
