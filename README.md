# cora_neural-network

## Simple explanation

The Cora dataset was not loading automatically.

So we made it simple:

1. Download the Cora files ourselves.
2. Save them in the folder PyTorch Geometric expects.
3. Load the dataset from those local files.

That is all [cora_setup.py](/Users/nils/PycharmProjects/cora_neural-network/cora_setup.py) does.

## Files

- [main.ipynb](/Users/nils/PycharmProjects/cora_neural-network/main.ipynb): the notebook you run
- [cora_setup.py](/Users/nils/PycharmProjects/cora_neural-network/cora_setup.py): the small helper that downloads and loads Cora

## What to say later

You can explain it like this:

"The automatic download failed, so we downloaded the Cora files manually, saved them in the right folder, and then loaded them normally."

## Quick test

If you want to test it from the shell, run:

```bash
./.venv/bin/python -c "from cora_setup import load_cora_dataset; dataset = load_cora_dataset(); print(dataset[0])"
```
