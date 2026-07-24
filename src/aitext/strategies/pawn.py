"""Strategy 2: token-level PAWN metrics (paper Section 3.2), evaluated with a single
80/20 holdout split (matches the original notebooks)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aitext.classifiers.evaluation import holdout_eval
from aitext.metrics.pawn import METRIC_NAMES
from aitext.models.base import ModelWrapper


def extract_features(model: ModelWrapper, texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame(model.pawn_metrics(texts))


def evaluate(
    features: pd.DataFrame,
    labels: np.ndarray,
    classifiers: list[str],
    seed: int = 42,
) -> list[dict]:
    X = features[METRIC_NAMES].values
    return [
        {"strategy": "pawn", "classifier": clf_name, **holdout_eval(clf_name, X, labels, seed=seed)}
        for clf_name in classifiers
    ]
