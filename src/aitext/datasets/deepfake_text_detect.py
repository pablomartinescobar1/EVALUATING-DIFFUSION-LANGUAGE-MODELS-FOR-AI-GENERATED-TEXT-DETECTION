"""DeepfakeTextDetect dataset loader (artem9k/ai-text-detection-pile, paper ref [23])."""
from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from aitext.datasets.base import balanced_sample, clean_text_column


def load(n_total: int, seed: int, split: str = "train") -> pd.DataFrame:
    dataset = load_dataset("artem9k/ai-text-detection-pile", split=split)
    df = pd.DataFrame(dataset)
    df["label"] = df["source"].apply(lambda source: 1 if source == "human" else 0)
    df = clean_text_column(df, text_col="text")
    return balanced_sample(df, label_col="label", n_total=n_total, seed=seed, positive_label=1)
