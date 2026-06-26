# CLAUDE.md — Cora GAT Experiment

## Research Question

> How do systematic variations in the number of attention heads (K) and hidden feature dimensions (F') impact node classification accuracy on the Cora dataset, and at what threshold does increased model capacity trigger overfitting?

**Status Quo:** The original GAT paper (Veličković et al., 2018) applies the model to Cora without explaining parameter decisions. This experiment identifies which parameters have significant impact on performance.

---

## Context

- **Course:** Digital Operations — Advanced Seminar, SS26
- **Assignment:** Application Exam (second assignment)
- **Author:** J. Lennart Hunsicker; Nils Becker
- **Presentation held:** Graph Attention Networks — theory, edge prediction (movies), graph prediction (drug discovery)

---

## The Cora Dataset

| Property | Value |
|---|---|
| Nodes | 2,708 scientific papers |
| Edges | 5,429 citations |
| Features per node | 1,433 (bag-of-words, binary) |
| Classes | 7 paper topics |
| Training nodes | 140 (20 per class) — very small |
| Validation nodes | 500 |
| Test nodes | 1,000 |

The training set is tiny relative to the graph. This makes Cora highly sensitive to overfitting as model capacity grows.

---

## GAT Architecture

Each GAT layer computes, for each node $i$:

```
e_ij  = LeakyReLU( a^T · [W·h_i || W·h_j] )   raw attention score
α_ij  = softmax over neighbors of e_ij          normalized weight
h_i'  = σ( Σ_{j∈N_i} α_ij · W · h_j )         aggregated update
```

With multi-head attention (K heads), intermediate layers concatenate head outputs → output dim = K × F'. The final layer averages heads to keep output dim = num_classes (7 for Cora).

**Original paper parameters (Veličković et al., 2018 — the baseline):**
- Layer 1: K=8 heads, F'=8 → 64-dim output (concat)
- Layer 2: K=1 head, F'=7 → 7-dim output (avg)
- Dropout: p=0.6 on input and between layers
- Optimizer: Adam, lr=0.005, weight_decay=5e-4
- Reported test accuracy: ~83%

---

## Current Implementation (`main.ipynb`)

The notebook has two models:

**GCN baseline** (standard graph convolution, no attention):
- 2 layers, hidden_channels=16, dropout=0.5
- Used as a reference point

**GAT implementation** (the focus of the experiment):
```python
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads)
        self.conv2 = GATConv(heads * hidden_channels, dataset.num_classes, heads)
```

**Known issue in current code:** `conv2` passes `heads` to the final layer, which with `concat=True` (PyG default) produces `heads * num_classes` output dimensions instead of `num_classes`. For the experiment, the final layer should use `concat=False` (averaging) to correctly output `num_classes=7`. Fix:
```python
self.conv2 = GATConv(heads * hidden_channels, dataset.num_classes, heads=1, concat=False)
```
Or equivalently keep multiple heads with averaging on the final layer.

**Data loading:** The automatic PyG download fails. `cora_setup.py` handles this by downloading raw files from GitHub and placing them where PyG expects. Run setup first.

---

## Experiment Design

### Primary Variables (research question core)

| Variable | Symbol | Original | Search space |
|---|---|---|---|
| Attention heads (layer 1) | K | 8 | 1, 2, 4, 8, 16 |
| Hidden dims per head | F' | 8 | 4, 8, 16, 32, 64 |
| Total representation dim | K × F' | 64 | 4 → 1,024 |

### Secondary Variables (extend analysis)

| Variable | Original | Search space | Notes |
|---|---|---|---|
| Dropout probability | 0.6 | 0.0, 0.3, 0.5, 0.6, 0.8 | Applied to input and between layers |
| Learning rate | 0.005 | 0.0005, 0.001, 0.005, 0.01 | Adam optimizer |
| L2 regularization (weight_decay) | 5e-4 | 0, 1e-4, 5e-4, 1e-3 | Second weight_decay term in Adam |
| Number of GAT layers | 2 | 1, 2, 3 | Risk of over-smoothing at 3+ |

### Attention Vector Dimension

In standard GATConv the attention vector `a ∈ ℝ^{2F'}` is always tied to F' — you cannot set it independently. As F' grows, so does the attention mechanism's capacity. This means F' controls **both** the feature embedding richness and the attention scoring capacity simultaneously. This coupling is worth noting as a limitation in the analysis.

### What to Measure

For each configuration:
- **Test accuracy** (primary metric)
- **Validation accuracy** (for early stopping / overfitting detection)
- **Train accuracy** (high train + low val = overfit)
- **Gap = train_acc − val_acc** (overfitting proxy)
- **Convergence epoch** (first epoch where val_acc no longer improves by >0.001)
- **Training time (s/epoch)** — benchmark wall-clock time per epoch, total time to convergence

### Training Time Benchmark

Add timing to the training loop to compare GAT configurations and GCN:

```python
import time

epoch_times = []
for epoch in range(1, 201):
    t0 = time.perf_counter()
    loss = train()
    epoch_times.append(time.perf_counter() - t0)

print(f"Mean time/epoch: {np.mean(epoch_times)*1000:.2f} ms")
print(f"Total training time: {sum(epoch_times):.2f} s")
```

Key drivers of training time on Cora:
- K × F' increases parameter count in W (K weight matrices of size F'×1433)
- Attention score computation scales with |E| × F' (linear in edges)
- Larger K×F' also increases memory bandwidth → measure both time and memory if GPU available

### Overfitting Threshold Hypothesis

With only 140 training nodes:
- **K×F' < 16** (e.g. 1×4): likely underfitting, insufficient capacity
- **K×F' ≈ 64** (8×8): original paper sweet spot, ~83% expected
- **K×F' > 256** (e.g. 16×32): overfitting expected — train_acc diverges from val_acc

The threshold is the inflection point in test accuracy as a function of K×F'. Report this as the "sweet spot" and justify it in relation to dataset size.

### Evaluation Protocol

Report mean ± std over **5 random seeds** per configuration (vary `torch.manual_seed`). Cora accuracy has high variance across seeds — single-run results are not reliable.

---

## Comparison: GAT vs GCN

The notebook already includes a GCN baseline. The analysis should produce a comparison table:

| Model | K | F' | K×F' | Test Acc (mean±std) | Train Time/epoch | Params |
|---|---|---|---|---|---|---|
| GCN | — | 16 | 16 | ? | ? | ? |
| GAT (paper) | 8 | 8 | 64 | ~0.83 | ? | ? |
| GAT (K=1, F'=8) | 1 | 8 | 8 | ? | ? | ? |
| GAT (K=4, F'=8) | 4 | 8 | 32 | ? | ? | ? |
| GAT (K=16, F'=8) | 16 | 8 | 128 | ? | ? | ? |
| GAT (K=8, F'=32) | 8 | 32 | 256 | ? | ? | ? |
| ... | | | | | | |

**GCN vs GAT key differences to discuss:**
- GCN uses fixed uniform neighbor averaging; GAT learns which neighbors matter
- GCN has fewer parameters (one shared W per layer); GAT has K weight matrices
- GCN is faster per epoch; GAT's attention overhead is O(|E|·F') extra
- Expected: GAT outperforms GCN on Cora because citation relevance is non-uniform

**Parameter count formulas:**
- GCN layer: `in_dim × out_dim` (e.g. 1433×16 = 22,928)
- GAT layer 1: `K × (in_dim × F') + K × 2F'` (W matrices + attention vectors)
  - Example K=8, F'=8: `8 × (1433×8) + 8×16 = 91,648`
- GAT layer 2 (avg, K=1): `(K_prev × F') × num_classes` (e.g. 64×7 = 448)

---

## Files

| File | Purpose |
|---|---|
| `main.ipynb` | Main notebook — GCN baseline + GAT implementation + training loop |
| `cora_setup.py` | Downloads Cora raw files and loads via PyG Planetoid |
| `data/Planetoid/Cora/` | Dataset files (raw + processed) |
| `gat_loss.png` | Saved accuracy curves from last run |

---

## Setup

```bash
# Create venv and install dependencies
python -m venv .venv
.venv/bin/pip install torch-geometric certifi scipy scikit-learn matplotlib

# Test data loading
.venv/bin/python -c "from cora_setup import load_cora_dataset; d = load_cora_dataset(); print(d[0])"

# Run notebook
.venv/bin/jupyter notebook main.ipynb
```

---

## Key GAT Facts (for reference in code/comments)

- Cora node features: 1433-dim binary bag-of-words
- Embedding step: `W ∈ ℝ^{F' × 1433}`, output `Wh_i ∈ ℝ^{F'}`
- Attention vector: `a ∈ ℝ^{2F'}` (dotted with concatenation of two F'-dim vectors)
- Non-neighbors: masked with −∞ so softmax assigns weight exactly 0
- Final layer: must use `concat=False` (average heads) to output 7-dim logits
- Over-smoothing risk: 2-layer GAT is standard; beyond 3 layers representations converge
- GAT complexity: O(|V|·F·F' + |E|·F') per layer — linear in edges, not V²
