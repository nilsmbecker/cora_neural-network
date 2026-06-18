from __future__ import annotations

import ssl
import urllib.request
import warnings
from pathlib import Path

import certifi
from torch_geometric.datasets import Planetoid

RAW_BASE_URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
RAW_FILE_NAMES = (
    "ind.cora.x",
    "ind.cora.tx",
    "ind.cora.allx",
    "ind.cora.y",
    "ind.cora.ty",
    "ind.cora.ally",
    "ind.cora.graph",
    "ind.cora.test.index",
)


def download_cora_raw(root: str = "data/Planetoid") -> Path:
    """Download the raw Cora files into the folder PyG expects."""
    raw_dir = Path(root) / "Cora" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    for file_name in RAW_FILE_NAMES:
        destination = raw_dir / file_name
        if destination.exists():
            continue

        source_url = f"{RAW_BASE_URL}/{file_name}"
        with urllib.request.urlopen(source_url, context=ssl_context) as response:
            destination.write_bytes(response.read())

    return raw_dir


def load_cora_dataset(root: str = "data/Planetoid", transform=None) -> Planetoid:
    """
    Simple flow:
    1. Download the raw Cora files.
    2. Put them where Planetoid expects them.
    3. Let Planetoid load the local files.
    """
    Planetoid.url = RAW_BASE_URL
    download_cora_raw(root=root)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
        )
        return Planetoid(root=root, name="Cora", transform=transform)
