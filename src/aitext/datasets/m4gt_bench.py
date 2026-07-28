"""M4GT-Bench dataset loader (paper reference [26]).

The official release (mbzuai-nlp/M4GT-Bench, and its SemEval-2024 Task 8
predecessor) only distributes data via a Google Drive folder + `gdown`, with no
official Hugging Face dataset -- so this loader uses the `d0rj/SemEval2024-task8`
community mirror instead, which exposes the same data as clean HF configs.

Scope: config `subtaskA_monolingual` (English-only binary human-vs-machine
detection), NOT `subtaskA_multilingual` (9 languages) -- deliberately narrowed to
stay consistent with the rest of this pipeline (tokenizers, PAWN masking, and the
zero-shot prompt are all English-only). See NOTES.md.

Label convention: this mirror uses label=0=human, label=1=machine -- the OPPOSITE of
this project's convention (label=1=human, 0=AI, used by every other loader).
`aitext.datasets.base.invert_binary_label` flips it; skipping that step would
silently invert every downstream ROC-AUC/accuracy number for this dataset.
"""
from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from aitext.datasets.base import balanced_sample, clean_text_column, invert_binary_label

_HF_ID = "d0rj/SemEval2024-task8"
_CONFIG = "subtaskA_monolingual"


def load(n_total: int, seed: int, split: str = "train") -> pd.DataFrame:
    dataset = load_dataset(_HF_ID, _CONFIG, split=split)
    df = invert_binary_label(dataset.to_pandas())
    df = clean_text_column(df, text_col="text")
    return balanced_sample(df, label_col="label", n_total=n_total, seed=seed, positive_label=1)
