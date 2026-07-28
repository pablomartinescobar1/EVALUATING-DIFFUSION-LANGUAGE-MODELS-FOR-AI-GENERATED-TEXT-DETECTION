"""Shared cloze-prompt mechanics for Strategy 4 (zero-shot classification), used by
all three paradigm wrappers -- mirrors why aitext.metrics.pawn centralizes PAWN math
instead of duplicating it per paradigm.

The prompt asks a fixed, minimal yes/no question and compares the model's own
log-probability for " Yes" vs " No" at one position. No classifier is ever fit on
this signal (see aitext.strategies.zero_shot) -- that is what makes it zero-shot.

Two correctness properties matter enough to be centralized here rather than
reimplemented per paradigm:

1. Truncation must happen at the id level, on the TEXT portion only, before the
   prompt/mask is appended. Tokenizing `text + prompt` as one string and letting
   `truncation=True` cut from the end (HF's default) would eat the prompt itself for
   any text long enough to need truncation -- common at max_length=512 for
   MAGE/RAID/DeepfakeTextDetect-sized inputs.
2. The mask position for MLM/diffusion models must be inserted as an already-resolved
   integer token id, never via `tokenizer.mask_token` string interpolation --
   DiffusionModel already needs a fallback (`vocab_size - 1`) for tokenizers where
   `mask_token_id` exists but the string form does not round-trip cleanly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

ZERO_SHOT_QUESTION = "Is the text above written by a human (not an AI)?"
YES_CANDIDATES = [" Yes", "Yes"]
NO_CANDIDATES = [" No", "No"]


def resolve_single_token_id(tokenizer, candidates: list[str]) -> int:
    """Return the id of the first candidate that tokenizes to exactly one token,
    trying the leading-space form first (GPT-2/RoBERTa/Llama-style BPE, where a token
    following a space is its own vocabulary entry) and the bare form second (BERT
    WordPiece). Raises rather than silently comparing a multi-token "word" via only
    its first sub-token, which would make the yes/no comparison meaningless.
    """
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(
        f"None of {candidates!r} is a single token for {tokenizer.__class__.__name__}."
    )


def build_causal_prompt_ids(tokenizer, text: str, max_length: int) -> list[int]:
    """Token ids for an autoregressive zero-shot prompt: (truncated) text ids followed
    by the fixed question, with the model's own special tokens added last. Truncation
    only ever shortens the text -- the question always survives intact."""
    prompt = f"\n\nQuestion: {ZERO_SHOT_QUESTION} Answer (Yes/No):"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    reserved = len(prompt_ids) + tokenizer.num_special_tokens_to_add(pair=False)
    budget = max(max_length - reserved, 1)
    text_ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=budget)["input_ids"]
    return tokenizer.build_inputs_with_special_tokens(text_ids + prompt_ids)


def build_cloze_input_ids(tokenizer, text: str, mask_token_id: int, max_length: int) -> list[int]:
    """Same idea for mask-predictor models (MLM / diffusion): inserts `mask_token_id`
    -- already resolved by the caller, e.g. DiffusionModel's fallback-aware
    self.mask_token_id -- directly as a token id, never via a `tokenizer.mask_token`
    string that may not exist for every trust_remote_code diffusion tokenizer."""
    prefix = f"\n\nQuestion: {ZERO_SHOT_QUESTION} Answer:"
    suffix = "."
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    reserved = len(prefix_ids) + 1 + len(suffix_ids) + tokenizer.num_special_tokens_to_add(pair=False)
    budget = max(max_length - reserved, 1)
    text_ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=budget)["input_ids"]
    core = text_ids + prefix_ids + [mask_token_id] + suffix_ids
    return tokenizer.build_inputs_with_special_tokens(core)


def last_real_position(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the rightmost non-pad position per row -- correct regardless of
    padding_side (unlike `attention_mask.sum(dim=1) - 1`, which silently assumes
    right-padding)."""
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return (attention_mask.float() * positions).argmax(dim=1)


def yes_no_log_odds(logits_at_answer: torch.Tensor, yes_token_id: int, no_token_id: int) -> list[float]:
    """logP(yes) - logP(no) at one position per row, given that position's raw
    logits (B, V). Positive => the model favors "Yes" => label=1=human under this
    project's convention, so no sign flip is needed downstream."""
    log_probs = F.log_softmax(logits_at_answer.float(), dim=-1)
    return (log_probs[:, yes_token_id] - log_probs[:, no_token_id]).cpu().tolist()
