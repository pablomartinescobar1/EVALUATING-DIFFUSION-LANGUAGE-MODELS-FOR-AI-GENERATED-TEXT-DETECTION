#!/usr/bin/env python
"""Aggregates results/tables/*.csv (produced by run_experiment.py) into the table
layouts used in the paper (Tables 2-7). Run after the relevant
configs/experiments/*.yaml have been executed.

    python scripts/build_paper_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "results" / "tables"

# Model -> paradigm, used to group/order rows the way the paper's tables do.
_PARADIGM_ORDER = {
    "gpt2": "Autoregressive",
    "gpt3_proxy": "Autoregressive",
    "llama7b": "Autoregressive",
    "bert": "Masked LM",
    "roberta": "Masked LM",
    "llada_8b_base": "Diffusion",
    "llada_8b_vrpo": "Diffusion",
    "dream_llada_7b": "Diffusion",
}

_MAIN_DATASETS = ["mage", "raid", "deepfaketextdetect"]


def _best_classifier_per_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, strategy), keep only the best classifier by ROC-AUC --
    matches how the paper's Tables 2-4 report "the best classifier per strategy"."""
    idx = df.groupby(["model", "strategy"])["roc_auc"].idxmax()
    best = df.loc[idx].copy()
    best["paradigm"] = best["model"].map(_PARADIGM_ORDER)
    best = best.sort_values(["paradigm", "model", "strategy"])
    return best[["paradigm", "model", "strategy", "classifier", "roc_auc", "accuracy", "f1"]]


def build_main_tables() -> None:
    """Tables 2 (MAGE), 3 (RAID), 4 (DeepfakeTextDetect)."""
    for dataset in _MAIN_DATASETS:
        path = _RESULTS_DIR / f"{dataset}.csv"
        if not path.exists():
            print(f"[skip] {path} not found -- run configs/experiments/{dataset}.yaml first.")
            continue
        df = pd.read_csv(path)
        table = _best_classifier_per_strategy(df)
        out_path = _RESULTS_DIR / f"{dataset}_paper_table.csv"
        table.to_csv(out_path, index=False)
        print(f"=== {dataset} ===")
        print(table.round(4).to_string(index=False))
        print(f"-> {out_path}\n")


def build_llada_family_table() -> None:
    """Table 5: parameter-controlled analysis (pivots model x dataset)."""
    path = _RESULTS_DIR / "llada_family.csv"
    if not path.exists():
        print(f"[skip] {path} not found -- run configs/experiments/llada_family.yaml first.")
        return
    df = pd.read_csv(path)
    best = df.loc[df.groupby(["dataset", "model", "strategy"])["roc_auc"].idxmax()]
    pivot = best.pivot_table(index=["model", "strategy"], columns="dataset", values="roc_auc")
    pivot = pivot.reindex(columns=[d for d in _MAIN_DATASETS if d in pivot.columns])
    out_path = _RESULTS_DIR / "llada_family_paper_table.csv"
    pivot.round(4).to_csv(out_path)
    print("=== llada_family (Table 5) ===")
    print(
        "NOTE: llada_8b_base and llada_8b_vrpo are BOTH 8B-parameter models "
        "(see configs/models.yaml / NOTES.md) -- this is not a small-vs-large-scale "
        "diffusion comparison, despite Section 5.2's current framing."
    )
    print(pivot.round(4).to_string())
    print(f"-> {out_path}\n")


def build_performance_table() -> None:
    """Table 7: inference time & peak VRAM (only non-cached extraction runs)."""
    perf_frames = []
    for perf_path in _RESULTS_DIR.glob("*_performance.csv"):
        frame = pd.read_csv(perf_path)
        frame["experiment"] = perf_path.stem.replace("_performance", "")
        perf_frames.append(frame)
    if not perf_frames:
        print("[skip] no *_performance.csv files found yet.")
        return
    combined = pd.concat(perf_frames, ignore_index=True)
    combined = combined[~combined["cached"]]
    out_path = _RESULTS_DIR / "performance_paper_table.csv"
    combined.to_csv(out_path, index=False)
    print("=== performance (Table 7) ===")
    print(combined.round(2).to_string(index=False))
    print(f"-> {out_path}\n")


def main() -> None:
    build_main_tables()
    build_llada_family_table()
    build_performance_table()


if __name__ == "__main__":
    main()
