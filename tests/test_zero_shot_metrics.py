"""Verifies aitext.metrics.zero_shot's pure scoring/token-resolution logic against
hand-checkable cases -- no GPU or model download needed."""
import pytest
import torch

from aitext.metrics.zero_shot import resolve_single_token_id, yes_no_log_odds


class _FakeTokenizer:
    """Minimal stand-in exposing only the one method resolve_single_token_id needs."""

    def __init__(self, encodings: dict[str, list[int]]):
        self._encodings = encodings

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._encodings[text]


def test_resolve_single_token_id_prefers_first_candidate_when_both_are_single_token():
    tokenizer = _FakeTokenizer({" Yes": [7], "Yes": [5]})
    assert resolve_single_token_id(tokenizer, [" Yes", "Yes"]) == 7


def test_resolve_single_token_id_falls_back_to_second_candidate():
    # " Yes" splits into 2 sub-tokens for this tokenizer, "Yes" (no leading space)
    # does not -- must fall back rather than returning a multi-token match.
    tokenizer = _FakeTokenizer({" Yes": [1, 2], "Yes": [5]})
    assert resolve_single_token_id(tokenizer, [" Yes", "Yes"]) == 5


def test_resolve_single_token_id_raises_when_no_candidate_is_single_token():
    tokenizer = _FakeTokenizer({" Yes": [1, 2], "Yes": [3, 4]})
    with pytest.raises(ValueError):
        resolve_single_token_id(tokenizer, [" Yes", "Yes"])


def test_yes_no_log_odds_equals_raw_logit_difference():
    # log_softmax(x)_yes - log_softmax(x)_no == x_yes - x_no exactly (the shared
    # logsumexp denominator cancels) -- true regardless of the other classes'
    # logits, which is exactly what these three rows exercise.
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    result = yes_no_log_odds(logits, yes_token_id=0, no_token_id=1)
    assert result[0] == pytest.approx(5.0, abs=1e-4)
    assert result[1] == pytest.approx(-5.0, abs=1e-4)
    assert result[2] == pytest.approx(0.0, abs=1e-4)
