"""Strategy 4: zero-shot cloze-prompt classification. No classifier is fit here --
the model's own Yes/No log-odds at a fixed prompt position (aitext.metrics.zero_shot)
IS the score. Evaluated on the FULL sample rather than a holdout split, since nothing
is fit to this dataset -- using every example gives a lower-variance estimate with no
overfitting risk to hold out against."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aitext.classifiers.evaluation import direct_score_eval
from aitext.models.base import ModelWrapper

FEATURE_COLUMNS = ["zero_shot_score"]


def extract_features(model: ModelWrapper, texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"zero_shot_score": model.zero_shot_score(texts)})


def evaluate(
    features: pd.DataFrame,
    labels: np.ndarray,
    classifiers: list[str],
    seed: int = 42,
) -> list[dict]:
    scores = features[FEATURE_COLUMNS].values[:, 0]
    metrics = direct_score_eval(scores, labels)
    # `classifiers`/`seed` are unused (no classifier is ever fit) but kept in the
    # signature for parity with every other strategy -- run_experiment() always calls
    # evaluate(features, labels, strategy_cfg["classifiers"], seed=seed). Each entry in
    # `classifiers` (normally just ["prompt"]) produces one identical row so the
    # (model, strategy) grouping in build_paper_tables.py needs no special-casing.
    return [{"strategy": "zero_shot", "classifier": name, **metrics} for name in classifiers]
