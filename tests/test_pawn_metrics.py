"""Verifies the unified PAWN metric formulas (aitext.metrics.pawn) against
hand-checkable toy logits -- no GPU or model download needed."""
import torch

from aitext.metrics.pawn import METRIC_NAMES, compute_pawn_metrics, sequence_log_likelihood


def test_pawn_metrics_top_token():
    # Single sequence, single position, vocab size 4. The occurred token (index 0)
    # is also the highest-logit token.
    logits = torch.tensor([[[2.0, 1.0, 0.0, 0.0]]])  # (B=1, T=1, V=4)
    labels = torch.tensor([[0]])
    weight_mask = torch.tensor([[1.0]])

    values = dict(zip(METRIC_NAMES, compute_pawn_metrics(logits, labels, weight_mask)))

    # The occurred token IS the argmax, so log-prob and max-log-prob must match.
    assert values["Mlog_prob"][0] == values["Mmax_log_prob"][0]
    # No token has strictly higher log-prob, so rank is 1 (best), normalized by V=4.
    assert abs(values["Mrank"][0] - 1 / 4) < 1e-5
    # top-p mass "at or above" the occurred token's probability is just its own mass.
    expected_top_p = torch.softmax(torch.tensor([2.0, 1.0, 0.0, 0.0]), dim=-1)[0].item()
    assert abs(values["Mtop_p"][0] - expected_top_p) < 1e-4
    assert values["Mentropy"][0] > 0  # non-degenerate distribution


def test_weight_mask_excludes_unmasked_positions():
    # Two positions, each confidently predicting its own occurred token (symmetric),
    # so restricting the mask to only position 0 must reproduce position 0's own value.
    logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])  # (B=1, T=2, V=2)
    labels = torch.tensor([[0, 1]])

    full_mask = torch.tensor([[1.0, 1.0]])
    first_only_mask = torch.tensor([[1.0, 0.0]])

    full_values = compute_pawn_metrics(logits, labels, full_mask)
    first_values = compute_pawn_metrics(logits, labels, first_only_mask)

    for metric_full, metric_first in zip(full_values, first_values):
        assert abs(metric_full[0] - metric_first[0]) < 1e-5


def test_sequence_log_likelihood_matches_manual_mean():
    logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    labels = torch.tensor([[0, 1]])
    mask = torch.tensor([[1.0, 1.0]])

    log_probs = torch.log_softmax(logits, dim=-1)
    expected = ((log_probs[0, 0, 0] + log_probs[0, 1, 1]) / 2).item()

    result = sequence_log_likelihood(logits, labels, mask)[0]
    assert abs(result - expected) < 1e-5
