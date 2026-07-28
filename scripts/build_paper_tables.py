#!/usr/bin/env python
"""Aggregates results/tables/*.csv (produced by run_experiment.py) into the table
layouts used in the paper. Run after the relevant configs/experiments/*.yaml have
been executed.

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
    "llama2_7b": "Autoregressive",
    "llama3_1_8b": "Autoregressive",
    "bert": "Masked LM",
    "roberta": "Masked LM",
    "modernbert": "Masked LM",
    "llada_8b_base": "Diffusion",
    "llada_8b_vrpo": "Diffusion",
    "dream_llada_7b": "Diffusion",
}

_MAIN_DATASETS = ["mage", "raid", "deepfaketextdetect", "beemo", "m4gt_bench"]

# Model groups for the cross-model comparison tables (see
# build_small_encoder_comparison_table/build_large_model_comparison_table below).
# NOT parameter-matched against each other -- there is no small-scale diffusion
# checkpoint in configs/models.yaml (see llada_8b_vrpo's own entry there, and
# NOTES.md), so these are two separate, internally-comparable groups rather than one
# "small vs large" pair.
_SMALL_ENCODER_MODELS = ["bert", "roberta", "modernbert"]
_LARGE_MODEL_COMPARISON_MODELS = ["llama2_7b", "llama3_1_8b", "llada_8b_base", "llada_8b_vrpo", "dream_llada_7b"]


def _best_classifier_per_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, strategy), keep only the best classifier by ROC-AUC --
    matches how the paper's per-dataset tables report "the best classifier per
    strategy"."""
    idx = df.groupby(["model", "strategy"])["roc_auc"].idxmax()
    best = df.loc[idx].copy()
    best["paradigm"] = best["model"].map(_PARADIGM_ORDER)
    best = best.sort_values(["paradigm", "model", "strategy"])
    return best[["paradigm", "model", "strategy", "classifier", "roc_auc", "accuracy", "f1"]]


def build_main_tables() -> None:
    """One per-dataset detection table for every entry in _MAIN_DATASETS (the
    paper's main results tables -- MAGE/RAID/DeepfakeTextDetect plus the
    advisor-requested Beemo/M4GT-Bench additions)."""
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


def _combined_main_results() -> pd.DataFrame | None:
    """Concatenates every already-built results/tables/<dataset>.csv from
    _MAIN_DATASETS (each already carries a `dataset` column, set by
    aitext.pipeline.run_experiment) into one frame, for aggregations that compare a
    chosen set of models across all main datasets at once. Returns None if none of
    those experiments have been run yet."""
    paths = [_RESULTS_DIR / f"{dataset}.csv" for dataset in _MAIN_DATASETS]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    return pd.concat(frames, ignore_index=True) if frames else None


def _pivot_best_by_model_dataset(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Restricts to `models`, keeps the best classifier per (dataset, model, strategy)
    by ROC-AUC, and pivots into (model, strategy) rows x dataset columns -- the shape
    used by the paper's cross-model comparison tables."""
    subset = df[df["model"].isin(models)]
    best = subset.loc[subset.groupby(["dataset", "model", "strategy"])["roc_auc"].idxmax()]
    pivot = best.pivot_table(index=["model", "strategy"], columns="dataset", values="roc_auc")
    pivot = pivot.reindex(columns=[d for d in _MAIN_DATASETS if d in pivot.columns])
    return pivot.round(4)


def build_llada_family_table() -> None:
    """Diffusion-family comparison (LLaDA-8B-Base vs VRPO vs Dream-LLaDA-7B), from
    configs/experiments/llada_family.yaml's own output -- a separate experiment run
    over mage/raid/deepfaketextdetect only, not part of _MAIN_DATASETS' per-dataset
    tables."""
    path = _RESULTS_DIR / "llada_family.csv"
    if not path.exists():
        print(f"[skip] {path} not found -- run configs/experiments/llada_family.yaml first.")
        return
    df = pd.read_csv(path)
    pivot = _pivot_best_by_model_dataset(df, models=list(df["model"].unique()))
    out_path = _RESULTS_DIR / "llada_family_paper_table.csv"
    pivot.to_csv(out_path)
    print("=== llada_family ===")
    print(
        "NOTE: llada_8b_base and llada_8b_vrpo are BOTH 8B-parameter models "
        "(see configs/models.yaml / NOTES.md) -- this is not a small-vs-large-scale "
        "diffusion comparison, despite the paper draft's current framing."
    )
    print(pivot.to_string())
    print(f"-> {out_path}\n")


def build_small_encoder_comparison_table() -> None:
    """Small-encoder comparison (BERT-base / RoBERTa-base / ModernBERT-base) across
    all _MAIN_DATASETS. NOT a parameter-matched "small-scale encoders vs small-scale
    diffusion" comparison -- no small-scale LLaDA checkpoint exists (NOTES.md;
    configs/models.yaml's llada_8b_vrpo entry), so this deliberately only compares the
    3 ~110-149M-parameter encoders against each other, across every strategy
    including the new zero-shot one."""
    combined = _combined_main_results()
    if combined is None:
        print("[skip] no main dataset tables found yet -- run the main experiments first.")
        return
    pivot = _pivot_best_by_model_dataset(combined, models=_SMALL_ENCODER_MODELS)
    out_path = _RESULTS_DIR / "small_encoder_comparison_paper_table.csv"
    pivot.to_csv(out_path)
    print("=== small encoder comparison (BERT / RoBERTa / ModernBERT) ===")
    print(pivot.to_string())
    print(f"-> {out_path}\n")


def build_large_model_comparison_table() -> None:
    """Large-model comparison (~7-8B parameters): LLaMA-2-7B / Llama-3.1-8B
    (autoregressive) vs. LLaDA-8B-Base / LLaDA-8B-VRPO / Dream-LLaDA-7B (diffusion)
    across all _MAIN_DATASETS -- unlike the small-encoder table above, this IS a
    genuinely parameter-comparable group (every model here is 7-8B)."""
    combined = _combined_main_results()
    if combined is None:
        print("[skip] no main dataset tables found yet -- run the main experiments first.")
        return
    pivot = _pivot_best_by_model_dataset(combined, models=_LARGE_MODEL_COMPARISON_MODELS)
    out_path = _RESULTS_DIR / "large_model_comparison_paper_table.csv"
    pivot.to_csv(out_path)
    print("=== large model comparison (~7-8B: autoregressive vs diffusion) ===")
    print(pivot.to_string())
    print(f"-> {out_path}\n")


def build_performance_table() -> None:
    """Inference time & peak VRAM per (dataset, model, strategy), only non-cached
    extraction runs."""
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
    print("=== performance (time & VRAM) ===")
    print(combined.round(2).to_string(index=False))
    print(f"-> {out_path}\n")


def main() -> None:
    build_main_tables()
    build_llada_family_table()
    build_small_encoder_comparison_table()
    build_large_model_comparison_table()
    build_performance_table()


if __name__ == "__main__":
    main()
