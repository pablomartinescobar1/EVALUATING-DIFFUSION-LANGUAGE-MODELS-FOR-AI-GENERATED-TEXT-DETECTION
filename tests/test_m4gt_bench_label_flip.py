"""Verifies aitext.datasets.base.invert_binary_label, which m4gt_bench.py uses to
convert the source mirror's native 0=human/1=machine convention to this project's
label=1=human/0=AI convention. Getting this backwards silently inverts every
downstream ROC-AUC/accuracy number for the dataset, so it earns a dedicated test."""
import pandas as pd

from aitext.datasets.base import invert_binary_label


def test_invert_binary_label_flips_native_convention():
    # Native mirror convention: 0=human, 1=machine.
    df = pd.DataFrame({"label": [0, 1, 0, 1], "text": ["h1", "m1", "h2", "m2"]})

    flipped = invert_binary_label(df)

    # This project's convention: 1=human, 0=machine/AI.
    assert flipped.loc[flipped["text"] == "h1", "label"].item() == 1
    assert flipped.loc[flipped["text"] == "m1", "label"].item() == 0
    assert flipped["label"].tolist() == [1, 0, 1, 0]


def test_invert_binary_label_does_not_mutate_input():
    df = pd.DataFrame({"label": [0, 1]})
    invert_binary_label(df)
    assert df["label"].tolist() == [0, 1]
