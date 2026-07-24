#!/usr/bin/env python
"""Grid-searches (mask_ratio, num_samples) for a diffusion model's Monte Carlo PAWN
extraction. Replaces the three near-duplicated notebooks that used to do this by hand
(LogisticRegresion+PawnMetrics.ipynb, MetricasFinales.ipynb, MorePatience.ipynb --
now archived in legacy/notebooks/). Meant for small-scale tuning runs on a few
hundred/thousand samples, not the final 10k-sample experiments -- use
scripts/run_experiment.py for those, with the winning (mask_ratio, num_samples)
written back into configs/models.yaml.

    python scripts/tune_diffusion_masking.py --model llada_8b_base --dataset mage \\
        --n-total 1000 --mask-ratios 0.25 0.3 0.35 0.4 --num-samples 7 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aitext.classifiers.evaluation import holdout_eval  # noqa: E402
from aitext.datasets.registry import get_dataset  # noqa: E402
from aitext.metrics.pawn import METRIC_NAMES  # noqa: E402
from aitext.models.registry import load_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Diffusion model name from configs/models.yaml")
    parser.add_argument("--dataset", default="mage")
    parser.add_argument("--n-total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=[0.25, 0.3, 0.35, 0.4])
    parser.add_argument("--num-samples", type=int, nargs="+", default=[7, 10])
    parser.add_argument("--classifier", default="deep_mlp")
    args = parser.parse_args()

    df = get_dataset(args.dataset, n_total=args.n_total, seed=args.seed)
    texts = df["text"].tolist()
    labels = df["label"].values

    model = load_model(args.model)  # loaded once, reused across the whole grid

    results = []
    for mask_ratio in args.mask_ratios:
        for num_samples in args.num_samples:
            model.mask_ratio = mask_ratio
            model.num_samples = num_samples

            features = pd.DataFrame(model.pawn_metrics(texts))
            metrics = holdout_eval(args.classifier, features[METRIC_NAMES].values, labels, seed=args.seed)
            results.append({"mask_ratio": mask_ratio, "num_samples": num_samples, **metrics})
            print(f"mask_ratio={mask_ratio} num_samples={num_samples} -> {metrics}")

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    out_path = _REPO_ROOT / "results" / "tables" / f"tune_{args.model}_{args.dataset}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nBest config:\n{results_df.iloc[0]}")
    print(f"Saved full grid to {out_path}")


if __name__ == "__main__":
    main()
