"""Dataset registry: reads configs/datasets.yaml and dispatches to the matching
loader module.

Loader modules are imported lazily (inside `get_dataset`), not at the top of this
file -- otherwise `import aitext.datasets` would require every dataset's own
dependency (the HF `datasets` package for mage.py/deepfake_text_detect.py, the `raid`
benchmark package for raid.py, ...) to be installed, even for someone who only wants
one of them. Adding a dataset = write a loader module returning a
DataFrame[text, label] (see mage.py/raid.py/deepfake_text_detect.py for the pattern),
add one line to `_LOADER_MODULES` below, and one entry to configs/datasets.yaml.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "datasets.yaml"

_LOADER_MODULES = {
    "mage": "aitext.datasets.mage",
    "raid": "aitext.datasets.raid",
    "deepfaketextdetect": "aitext.datasets.deepfake_text_detect",
}


def _load_registry() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def list_datasets() -> list[str]:
    return list(_load_registry().keys())


def get_dataset(name: str, n_total: int, seed: int) -> pd.DataFrame:
    registry = _load_registry()
    if name not in registry:
        raise KeyError(f"Dataset '{name}' is not registered in {_CONFIG_PATH}")
    entry = registry[name]
    loader_name = entry["loader"]
    if loader_name not in _LOADER_MODULES:
        raise ValueError(f"No loader implemented for '{loader_name}'")

    loader_module = importlib.import_module(_LOADER_MODULES[loader_name])
    return loader_module.load(n_total=n_total, seed=seed, split=entry["split"])
