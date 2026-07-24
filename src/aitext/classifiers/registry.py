"""Classifier registry shared by every strategy.

Factory modules are imported lazily (inside `make_classifier`), not at the top of
this file -- otherwise `import aitext.classifiers.deep_mlp` (torch only) would also
require xgboost/scikit-learn's tree dependencies to be installed just to load the
registry. Adding a classifier: write a `factory(seed) -> object` returning something
with `.fit(X, y)` and `.predict_proba(X)` (see linear.py/trees.py/deep_mlp.py), add
one entry to `_FACTORY_PATHS` below, and reference its name from any experiment
YAML's `classifiers:` list. No strategy code needs to change.
"""
from __future__ import annotations

import importlib

_FACTORY_PATHS: dict[str, tuple[str, str]] = {
    "logreg": ("aitext.classifiers.linear", "make_logreg"),
    "random_forest": ("aitext.classifiers.trees", "make_random_forest"),
    "xgboost": ("aitext.classifiers.trees", "make_xgboost"),
    "xgboost_calibrated": ("aitext.classifiers.trees", "make_xgboost_calibrated"),
    "deep_mlp": ("aitext.classifiers.deep_mlp", "make_deep_mlp"),
}


def list_classifiers() -> list[str]:
    return list(_FACTORY_PATHS.keys())


def make_classifier(name: str, seed: int = 42):
    if name not in _FACTORY_PATHS:
        raise KeyError(f"Classifier '{name}' is not registered. Known: {list_classifiers()}")
    module_path, factory_name = _FACTORY_PATHS[name]
    factory = getattr(importlib.import_module(module_path), factory_name)
    return factory(seed)
