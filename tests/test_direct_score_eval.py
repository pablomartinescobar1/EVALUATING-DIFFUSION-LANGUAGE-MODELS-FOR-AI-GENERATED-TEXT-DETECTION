"""Verifies aitext.classifiers.evaluation.direct_score_eval, the Strategy 4
(zero-shot) harness that evaluates a raw signed score with no fit and no
train/test split."""
import numpy as np

from aitext.classifiers.evaluation import direct_score_eval


def test_direct_score_eval_perfect_separation():
    scores = np.array([5.0, 3.0, -2.0, -4.0])
    labels = np.array([1, 1, 0, 0])

    metrics = direct_score_eval(scores, labels)

    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_direct_score_eval_thresholds_at_zero_not_half():
    # zero_shot_score is a log-odds difference (can be any sign/magnitude), not a
    # calibrated [0,1] probability -- a score of 0.3 must already count as
    # label=1="human" even though it is far below the 0.5 threshold Strategies 1-3's
    # calibrated-probability harnesses use. This is exactly the bug a copy-pasted
    # `_evaluate_probs(..., threshold=0.5)` call would reintroduce.
    scores = np.array([0.3, -0.3])
    labels = np.array([1, 0])

    metrics = direct_score_eval(scores, labels)

    assert metrics["accuracy"] == 1.0
