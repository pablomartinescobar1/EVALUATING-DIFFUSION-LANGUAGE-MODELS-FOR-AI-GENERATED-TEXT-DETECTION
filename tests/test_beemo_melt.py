"""Verifies aitext.datasets.base.melt_binary_text_columns, which beemo.py uses to
turn Beemo's wide source table (a human reference and a machine output as sibling
columns in the same row) into this project's usual one-row-per-text long format."""
import pandas as pd

from aitext.datasets.base import melt_binary_text_columns


def test_melt_produces_two_rows_per_source_row_with_correct_labels():
    df = pd.DataFrame(
        {
            "human_output": ["human text A", "human text B"],
            "model_output": ["ai text A", "ai text B"],
            "unused_column": ["x", "y"],
        }
    )

    melted = melt_binary_text_columns(df, positive_col="human_output", negative_col="model_output")

    assert len(melted) == 4
    assert set(melted.columns) == {"text", "label"}

    human_rows = melted[melted["label"] == 1]
    ai_rows = melted[melted["label"] == 0]
    assert sorted(human_rows["text"]) == ["human text A", "human text B"]
    assert sorted(ai_rows["text"]) == ["ai text A", "ai text B"]
