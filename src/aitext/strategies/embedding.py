"""Strategy 3: embedding-based classification (paper Section 3.3), evaluated with a
single 80/20 holdout split (matches the original notebooks)."""
from __future__ import annotations

import numpy as np

from aitext.classifiers.evaluation import holdout_eval
from aitext.models.base import ModelWrapper


def extract_features(model: ModelWrapper, texts: list[str]) -> np.ndarray:
    return model.embed(texts)


def evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    classifiers: list[str],
    seed: int = 42,
) -> list[dict]:
    return [
        {"strategy": "embedding", "classifier": clf_name, **holdout_eval(clf_name, features, labels, seed=seed)}
        for clf_name in classifiers
    ]
