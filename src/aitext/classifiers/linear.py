"""Logistic regression factory, shared across all 3 strategies."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def make_logreg(seed: int = 42) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
