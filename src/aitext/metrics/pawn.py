"""Token-level PAWN metrics (paper Eqs. 2-6): log-probability, entropy, max log-probability,
normalized rank, and top-p mass of the occurred token.

The original notebooks had three near-identical copies of this math
(`calculate_five_metrics`, `calculate_five_metrics_mlm`, `calculate_five_metrics_diffusion`):
they differed only in *which positions* get averaged into the per-sequence score --
all non-pad positions for autoregressive/MLM scoring, or only the positions the model
had to reconstruct for diffusion masking. That's exactly what `weight_mask` below
parametrizes, so one function now covers all three cases.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

METRIC_NAMES = ["Mlog_prob", "Mentropy", "Mmax_log_prob", "Mrank", "Mtop_p"]


def compute_pawn_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight_mask: torch.Tensor,
) -> list[list[float]]:
    """Compute the 5 PAWN token-level metrics, averaged per sequence over the
    positions where `weight_mask` is nonzero.

    Args:
        logits: (B, T, V) raw model logits at each position.
        labels: (B, T) token ids the metrics are evaluated against at each position --
            the *next* token for shifted autoregressive-style logits, or the
            *original* token for MLM / diffusion reconstruction-style logits.
        weight_mask: (B, T) mask selecting which positions count towards the
            per-sequence average (attention mask for AR/MLM scoring, masked
            positions for diffusion PAWN).

    Returns:
        A list of 5 lists (one per metric, in `METRIC_NAMES` order), each of
        length B (one value per sequence in the batch).
    """
    _, _, vocab_size = logits.shape
    weight_mask = weight_mask.float()

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    log_p = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    max_log_p, _ = log_probs.max(dim=-1)
    rank = (log_probs > log_p.unsqueeze(-1)).sum(dim=-1).float() + 1.0
    rank_norm = rank / vocab_size
    prob_occurred = probs.gather(2, labels.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)
    top_p = (probs * (probs >= prob_occurred)).sum(dim=-1)

    weight_sum = weight_mask.sum(dim=1).clamp(min=1)
    results = []
    for metric in (log_p, entropy, max_log_p, rank_norm, top_p):
        avg = (metric * weight_mask).sum(dim=1) / weight_sum
        results.append(avg.cpu().tolist())
    return results


def sequence_log_likelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight_mask: torch.Tensor,
) -> list[float]:
    """Strategy-1 global score: per-sequence mean token log-likelihood, i.e. Eq. 1's
    -mean(loss) computed per example (NOT via the model's batch-averaged `.loss`,
    which is the bug this refactor fixes -- see NOTES.md).
    """
    weight_mask = weight_mask.float()
    log_probs = F.log_softmax(logits, dim=-1)
    log_p = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    weight_sum = weight_mask.sum(dim=1).clamp(min=1)
    return ((log_p * weight_mask).sum(dim=1) / weight_sum).cpu().tolist()
