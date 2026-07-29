"""Orchestrates one experiment config end-to-end: load a dataset, then for each
model and each strategy, extract features (cached to disk) and classify.

This single function replaces the ad hoc, copy-pasted "Scores / PAWN / Embeddings /
Clasificación" sections that used to be duplicated across
FinalResultsMAGE/RAID/DeepfakeTextDetect.ipynb (and, for the small-scale diffusion
comparison, FinalResultsLLADAs.ipynb).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from aitext.datasets.registry import get_dataset
from aitext.models.registry import load_model
from aitext.resource_tracking import resource_wrapper
from aitext.strategies import STRATEGY_MODULES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FEATURES_DIR = _REPO_ROOT / "results" / "features"


def load_experiment_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _feature_cache_path(
    dataset_name: str, model_name: str, strategy_name: str, n_total: int, seed: int
) -> Path:
    """`n_total`/`seed` are part of the cache key, not just the dataset/model/strategy
    name: they determine WHICH texts `balanced_sample` draws, so a cached file from a
    different `n_total` (e.g. an ad hoc smaller run) or `seed` has a different length
    or different underlying texts entirely. Without this, loading a stale cache from
    a smaller/differently-seeded run silently mismatches the current run's `labels`
    array -- at best a length-mismatch crash downstream, at worst (same length,
    different seed) silently misaligned texts and labels with no error at all."""
    directory = _FEATURES_DIR / dataset_name
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "npy" if strategy_name == "embedding" else "csv"
    return directory / f"{model_name}_{strategy_name}_n{n_total}_s{seed}.{suffix}"


def _save_features(cache_path: Path, strategy_name: str, features) -> None:
    if strategy_name == "embedding":
        np.save(cache_path, features)
    else:
        features.to_csv(cache_path, index=False)


def _load_cached_features(cache_path: Path, strategy_name: str):
    if strategy_name == "embedding":
        return np.load(cache_path)
    return pd.read_csv(cache_path)


def _extract_or_load_features(
    model,
    model_name: str,
    dataset_name: str,
    strategy_name: str,
    texts: list[str],
    n_total: int,
    seed: int,
) -> tuple[Any, float, float, float, bool]:
    cache_path = _feature_cache_path(dataset_name, model_name, strategy_name, n_total, seed)
    if cache_path.exists():
        return _load_cached_features(cache_path, strategy_name), 0.0, 0.0, 0.0, True

    strategy_module = STRATEGY_MODULES[strategy_name]
    features, elapsed, vram_peak, cpu_delta = resource_wrapper(
        strategy_module.extract_features, model, texts, device=model.device
    )
    _save_features(cache_path, strategy_name, features)
    return features, elapsed, vram_peak, cpu_delta, False


def run_experiment(config: dict) -> pd.DataFrame:
    dataset_names = config.get("datasets") or [config["dataset"]]
    n_total = config["n_total"]
    seed = config["seed"]
    max_length = config.get("max_length", 512)
    batch_size = config.get("batch_size", 4)
    strategies_config = config["strategies"]

    result_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []

    for dataset_name in dataset_names:
        df = get_dataset(dataset_name, n_total=n_total, seed=seed)
        texts = df["text"].tolist()
        labels = df["label"].values

        for model_name in config["models"]:
            model = load_model(model_name, max_length=max_length, batch_size=batch_size)

            for strategy_name, strategy_cfg in strategies_config.items():
                features, elapsed, vram_peak, cpu_delta, cached = _extract_or_load_features(
                    model, model_name, dataset_name, strategy_name, texts, n_total, seed
                )
                performance_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "strategy": strategy_name,
                        "time_seconds": elapsed,
                        "vram_peak_gb": vram_peak,
                        "cpu_ram_delta_gb": cpu_delta,
                        "cached": cached,
                    }
                )

                strategy_module = STRATEGY_MODULES[strategy_name]
                rows = strategy_module.evaluate(features, labels, strategy_cfg["classifiers"], seed=seed)
                for row in rows:
                    result_rows.append({"dataset": dataset_name, "model": model_name, **row})

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results_df = pd.DataFrame(result_rows)
    output_path = _REPO_ROOT / config["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    performance_df = pd.DataFrame(performance_rows)
    performance_path = output_path.with_name(output_path.stem + "_performance.csv")
    performance_df.to_csv(performance_path, index=False)

    return results_df
