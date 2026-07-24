"""Regression test for the batch-averaged-loss bug this refactor fixes (see
NOTES.md item 1): the original `batch_autoregressive_scores` read HF's
`outputs.loss` -- already averaged across the whole batch -- and repeated it once
per example, so every text in a batch of 4 got an identical score. This test proves
`sequence_log_likelihood` (the replacement) keeps each example's score independent
of what else is in its batch, using pure synthetic logits -- no GPU/model needed.
"""
import torch

from aitext.metrics.pawn import sequence_log_likelihood


def test_scores_are_independent_per_example():
    # Example A: model confidently predicts the occurred token everywhere -> high score.
    logits_confident = torch.zeros(1, 3, 5)
    logits_confident[0, :, 0] = 10.0
    labels_confident = torch.zeros(1, 3, dtype=torch.long)

    # Example B: uniform logits -> the occurred token is no more likely than chance.
    logits_uniform = torch.zeros(1, 3, 5)
    labels_uniform = torch.zeros(1, 3, dtype=torch.long)

    mask = torch.ones(1, 3)

    score_confident = sequence_log_likelihood(logits_confident, labels_confident, mask)[0]
    score_uniform = sequence_log_likelihood(logits_uniform, labels_uniform, mask)[0]
    assert score_confident > score_uniform

    # Batched together, each example must retain ITS OWN value -- exactly what the
    # original `.repeat(len(batch))` bug violated (both would have collapsed to the
    # batch mean regardless of their individual content).
    batched_logits = torch.cat([logits_confident, logits_uniform], dim=0)
    batched_labels = torch.cat([labels_confident, labels_uniform], dim=0)
    batched_mask = torch.cat([mask, mask], dim=0)

    batched_scores = sequence_log_likelihood(batched_logits, batched_labels, batched_mask)

    assert batched_scores[0] != batched_scores[1]
    assert abs(batched_scores[0] - score_confident) < 1e-5
    assert abs(batched_scores[1] - score_uniform) < 1e-5
