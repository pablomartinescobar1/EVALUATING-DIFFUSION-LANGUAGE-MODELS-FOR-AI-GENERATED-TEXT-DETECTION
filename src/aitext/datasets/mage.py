"""MAGE dataset loader (yaful/MAGE on Hugging Face, paper reference [22])."""
from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from aitext.datasets.base import balanced_sample, clean_text_column


def load(n_total: int, seed: int, split: str = "test") -> pd.DataFrame:
    dataset = load_dataset("yaful/MAGE", split=split)
    df = dataset.to_pandas()
    df = clean_text_column(df, text_col="text")
    # MAGE convention (kept from the original notebooks): label == 1 is human, 0 is AI.
    return balanced_sample(df, label_col="label", n_total=n_total, seed=seed, positive_label=1)
