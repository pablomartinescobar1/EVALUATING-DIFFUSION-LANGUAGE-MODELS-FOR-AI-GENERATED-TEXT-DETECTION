"""Shared dataset utilities: text cleanup, wide-to-long melting, and balanced
sampling, used by every dataset loader so the 50/50 human/AI balance guarantee is
identical across datasets.

This is what fixes the original MAGE notebook, which only did `.sample(n=10000,
random_state=42)` on the unbalanced full split and merely *reported* the resulting
class balance afterwards, unlike RAID/DeepfakeTextDetect which already stratified
explicitly. See NOTES.md.

Kept dependency-light (pandas only) so it can be unit tested without installing the
HF `datasets` package that individual loaders (mage.py/raid.py/beemo.py/...) need.
"""
from __future__ import annotations

import pandas as pd


def clean_text_column(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df["text"] = df[text_col].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    return df


def melt_binary_text_columns(df: pd.DataFrame, positive_col: str, negative_col: str) -> pd.DataFrame:
    """Turns a WIDE row -- one column of label=1 texts, one column of label=0 texts,
    both for the same underlying example (e.g. Beemo's `human_output`/`model_output`
    sibling columns) -- into this project's usual long format: one row per text with
    a binary `label` column, ready for `clean_text_column`/`balanced_sample`."""
    positive = df[[positive_col]].rename(columns={positive_col: "text"})
    positive["label"] = 1
    negative = df[[negative_col]].rename(columns={negative_col: "text"})
    negative["label"] = 0
    return pd.concat([positive, negative], ignore_index=True)


def invert_binary_label(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Flips a 0/1 label column in place (via a copy). Some source datasets (e.g.
    M4GT-Bench's `d0rj/SemEval2024-task8` mirror) natively use 0=human/1=machine --
    the opposite of this project's label=1=human/0=AI convention -- and need this
    before `balanced_sample`'s `positive_label=1` default is meaningful."""
    df = df.copy()
    df[label_col] = 1 - df[label_col]
    return df


def balanced_sample(
    df: pd.DataFrame,
    label_col: str,
    n_total: int,
    seed: int,
    positive_label=1,
) -> pd.DataFrame:
    """Sample exactly `n_total // 2` rows of each class, shuffle, and reset the index.

    Raises rather than silently returning fewer rows if either class is too small --
    a loud failure here is much cheaper to debug than a silently-imbalanced dataset
    feeding into every downstream experiment.
    """
    n_per_class = n_total // 2
    positives = df[df[label_col] == positive_label]
    negatives = df[df[label_col] != positive_label]

    if len(positives) < n_per_class or len(negatives) < n_per_class:
        raise ValueError(
            f"Not enough rows to balance {n_total} total: "
            f"{len(positives)} positive, {len(negatives)} negative, need {n_per_class} each."
        )

    sampled_positive = positives.sample(n=n_per_class, random_state=seed, replace=False)
    sampled_negative = negatives.sample(n=n_per_class, random_state=seed, replace=False)

    balanced = pd.concat([sampled_positive, sampled_negative])
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    balanced["text_id"] = balanced.index
    return balanced
