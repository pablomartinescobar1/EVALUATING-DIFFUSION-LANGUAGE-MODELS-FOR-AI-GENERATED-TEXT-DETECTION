"""Common interface every model backbone implements, regardless of paradigm
(autoregressive / masked_lm / diffusion).

Strategies (src/aitext/strategies/) call only these four methods and never need to
know which paradigm they're talking to -- that's what lets e.g. `strategies/embedding.py`
run against *every* configured model through one shared code path, instead of the ad hoc
per-model loops the original notebooks had (which is exactly how GPT-3 silently got
skipped from the embeddings loop in FinalResultsMAGE.ipynb -- see NOTES.md).
"""
from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm


class ModelWrapper(ABC):
    """Wraps a loaded tokenizer+model pair and exposes the 4 detection strategies."""

    name: str
    max_length: int = 512
    batch_size: int = 4

    @abstractmethod
    def score(self, texts: list[str]) -> list[float]:
        """Strategy 1: global per-sequence log-likelihood score (paper Eq. 1)."""

    @abstractmethod
    def pawn_metrics(self, texts: list[str]) -> dict[str, list[float]]:
        """Strategy 2: the 5 PAWN token-level metrics, keyed by
        aitext.metrics.pawn.METRIC_NAMES."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Strategy 3: one embedding vector per text, shape (len(texts), hidden_dim)."""

    @abstractmethod
    def zero_shot_score(self, texts: list[str]) -> list[float]:
        """Strategy 4: logP(" Yes") - logP(" No") at a fixed cloze-prompt position
        asking whether the text was written by a human (see aitext.metrics.zero_shot).
        Higher => model favors "human", matching label=1=human, so no sign flip is
        needed downstream. No classifier is fit on this (see aitext.strategies.zero_shot)."""


def iter_batches(texts: list[str], batch_size: int, desc: str | None = None) -> Iterator[list[str]]:
    indices = range(0, len(texts), batch_size)
    if desc is not None:
        indices = tqdm(indices, desc=desc)
    for i in indices:
        yield texts[i : i + batch_size]


def autocast_context(device: str, dtype: torch.dtype):
    """`torch.amp.autocast` on CUDA, a no-op context on CPU (so wrappers also run,
    slowly, in a GPU-less environment -- e.g. for unit tests with a dummy model)."""
    if device == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()
