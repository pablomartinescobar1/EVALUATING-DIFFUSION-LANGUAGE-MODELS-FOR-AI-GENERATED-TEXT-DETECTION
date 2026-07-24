"""Verifies aitext.datasets.base.balanced_sample gives an exact 50/50 split -- this
is what fixes the original MAGE loader, which never guaranteed class balance the way
RAID/DeepfakeTextDetect already did (see NOTES.md item 3)."""
import pandas as pd
import pytest

from aitext.datasets.base import balanced_sample


def test_balanced_sample_exact_split():
    df = pd.DataFrame(
        {
            "label": [1] * 700 + [0] * 300,
            "text": [f"text-{i}" for i in range(1000)],
        }
    )
    result = balanced_sample(df, label_col="label", n_total=400, seed=0)

    assert len(result) == 400
    counts = result["label"].value_counts()
    assert counts[1] == 200
    assert counts[0] == 200
    # No duplicate rows introduced by sampling.
    assert result["text"].nunique() == 400


def test_balanced_sample_raises_when_class_too_small():
    df = pd.DataFrame({"label": [1] * 10 + [0] * 500, "text": [f"t{i}" for i in range(510)]})
    with pytest.raises(ValueError):
        balanced_sample(df, label_col="label", n_total=100, seed=0)  # needs 50 positives, only 10 exist
