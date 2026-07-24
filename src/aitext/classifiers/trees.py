"""Tree-based classifier factories (Strategy 1: global score, paper Tables 2-4)."""
from __future__ import annotations

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


def make_random_forest(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, random_state=seed)


def make_xgboost(seed: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, eval_metric="logloss", random_state=seed)


def make_xgboost_calibrated(seed: int = 42) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        xgb.XGBClassifier(eval_metric="logloss", random_state=seed), method="isotonic", cv=3
    )
