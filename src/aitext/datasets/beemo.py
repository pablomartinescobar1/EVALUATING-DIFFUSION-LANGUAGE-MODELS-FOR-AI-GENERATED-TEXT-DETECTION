"""Beemo dataset loader (toloka/beemo on Hugging Face, paper reference [27]).

Unlike mage.py/raid.py/deepfake_text_detect.py, the source table is WIDE: each row
pairs one human reference (`human_output`) with one machine output (`model_output`)
as sibling columns for the same prompt, plus expert/LLM-edited variants
(`human_edits`, `llama-3.1-70b_edits`, `gpt-4o_edits`) that represent mixed
human/AI authorship rather than a clean binary label -- not used here, since this
project's binary detection task only needs one human column and one AI column.
`aitext.datasets.base.melt_binary_text_columns` turns it into this project's usual
long format.
"""
from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from aitext.datasets.base import balanced_sample, clean_text_column, melt_binary_text_columns


def load(n_total: int, seed: int, split: str = "train") -> pd.DataFrame:
    dataset = load_dataset("toloka/beemo", split=split)
    df = melt_binary_text_columns(dataset.to_pandas(), positive_col="human_output", negative_col="model_output")
    df = clean_text_column(df, text_col="text")
    return balanced_sample(df, label_col="label", n_total=n_total, seed=seed, positive_label=1)
