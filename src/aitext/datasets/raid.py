"""RAID dataset loader (paper reference [16]), via the `raid` benchmark package
(https://github.com/liamdugan/raid -- install per that repo's instructions)."""
from __future__ import annotations

import pandas as pd
from raid.utils import load_data

from aitext.datasets.base import balanced_sample, clean_text_column


def load(n_total: int, seed: int, split: str = "train") -> pd.DataFrame:
    df = load_data(split=split).copy()
    df["text"] = df["generation"]
    df["label"] = df["model"].apply(lambda model: 1 if model == "human" else 0)
    df = clean_text_column(df, text_col="text")
    return balanced_sample(df, label_col="label", n_total=n_total, seed=seed, positive_label=1)
