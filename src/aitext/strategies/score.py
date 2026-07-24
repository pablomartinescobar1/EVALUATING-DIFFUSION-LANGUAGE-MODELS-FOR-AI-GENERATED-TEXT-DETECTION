"""Strategy 1: global log-likelihood score (paper Eq. 1, Section 3.1).

Evaluated via 5-fold cross-validation on the single score column -- matches how the
original notebooks evaluated this strategy (unlike PAWN/Embedding, which use a single
holdout split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aitext.classifiers.evaluation import cross_validated_eval
from aitext.models.base import ModelWrapper

FEATURE_COLUMNS = ["score"]


def extract_features(model: ModelWrapper, texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"score": model.score(texts)})


def evaluate(
    features: pd.DataFrame,
    labels: np.ndarray,
    classifiers: list[str],
    seed: int = 42,
) -> list[dict]:
    X = features[FEATURE_COLUMNS].values
    return [
        {"strategy": "score", "classifier": clf_name, **cross_validated_eval(clf_name, X, labels, seed=seed)}
        for clf_name in classifiers
    ]
