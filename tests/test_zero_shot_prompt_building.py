"""Regression tests for the two correctness properties aitext.metrics.zero_shot's
prompt-builders exist to guarantee (see that module's docstring): truncation only
ever shortens the TEXT, never the fixed question/mask, and the mask/question always
survives intact even when the input text is far longer than max_length.

Uses a minimal word-level fake tokenizer (each whitespace-separated word is one
token id) instead of a real one -- a long repeated-word text then produces far more
"tokens" than any max_length used here, while the short, fixed English templates stay
well within budget, without needing a real tokenizer/model download.
"""
from __future__ import annotations

from aitext.metrics.zero_shot import ZERO_SHOT_QUESTION, build_causal_prompt_ids, build_cloze_input_ids


class _FakeTokenizer:
    def __init__(self, num_special: int = 2):
        self._vocab: dict[str, int] = {}
        self._num_special = num_special

    def _ids_for(self, text: str) -> list[int]:
        return [self._vocab.setdefault(word, len(self._vocab)) for word in text.split()]

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = self._ids_for(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self._num_special

    def build_inputs_with_special_tokens(self, ids: list[int]) -> list[int]:
        return [-1, *ids, -2]  # distinguishable BOS/EOS markers


def test_build_causal_prompt_ids_preserves_prompt_when_text_is_long():
    tokenizer = _FakeTokenizer()
    prompt = f"\n\nQuestion: {ZERO_SHOT_QUESTION} Answer (Yes/No):"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    max_length = len(prompt_ids) + 10  # room for the prompt plus a few text words

    ids = build_causal_prompt_ids(tokenizer, "word " * 10_000, max_length)

    assert ids[0] == -1 and ids[-1] == -2
    core = ids[1:-1]
    assert core[-len(prompt_ids) :] == prompt_ids
    assert len(ids) <= max_length


def test_build_causal_prompt_ids_keeps_short_text_whole():
    tokenizer = _FakeTokenizer()
    short_text = "a short human written sentence"
    text_ids = tokenizer(short_text)["input_ids"]

    ids = build_causal_prompt_ids(tokenizer, short_text, max_length=512)

    core = ids[1:-1]
    assert core[: len(text_ids)] == text_ids


def test_build_cloze_input_ids_preserves_mask_and_suffix_when_text_is_long():
    tokenizer = _FakeTokenizer()
    mask_id = 999
    prefix = f"\n\nQuestion: {ZERO_SHOT_QUESTION} Answer:"
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(".", add_special_tokens=False)["input_ids"]
    max_length = len(prefix_ids) + 1 + len(suffix_ids) + 10

    ids = build_cloze_input_ids(tokenizer, "word " * 10_000, mask_id, max_length)

    core = ids[1:-1]
    tail = prefix_ids + [mask_id] + suffix_ids
    assert core[-len(tail) :] == tail
    assert len(ids) <= max_length
