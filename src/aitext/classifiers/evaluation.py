"""Two reusable evaluation harnesses shared by every strategy:

- `cross_validated_eval`: 5-fold StratifiedKFold cross-validation, used by Strategy 1
  (global score) -- matches how the original notebooks evaluated that strategy.
- `holdout_eval`: a single 80/20 stratified split with StandardScaler, used by
  Strategy 2 (PAWN) and Strategy 3 (embeddings) -- matches the original notebooks'
  single train_test_split + StandardScaler + fit/eval.

Keeping these as two shared functions (instead of duplicating the split/scale/fit/
score boilerplate inside every strategy module, as the original notebooks did 3-6
times each) is what makes adding a new strategy or classifier a small, local change.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

from aitext.classifiers.registry import make_classifier

METRIC_KEYS = ["roc_auc", "accuracy", "f1", "recall", "precision"]


def _evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else float("nan"),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
    }


def cross_validated_eval(
    clf_name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    cv_folds: int = 5,
) -> dict[str, float]:
    clf = make_classifier(clf_name, seed=seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_results = cross_validate(clf, X, y, cv=cv, scoring=METRIC_KEYS)
    return {key: float(np.mean(cv_results[f"test_{key}"])) for key in METRIC_KEYS}


def holdout_eval(
    clf_name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    test_size: float = 0.2,
    scale: bool = True,
) -> dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = make_classifier(clf_name, seed=seed)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return _evaluate_probs(y_test, y_prob)
