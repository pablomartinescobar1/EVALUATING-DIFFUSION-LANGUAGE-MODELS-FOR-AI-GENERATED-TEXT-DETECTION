"""Regression test for the cache-key bug this fixes: aitext.pipeline._feature_cache_path
used to key the on-disk feature cache by (dataset, model, strategy) only, so re-running
the same dataset/model/strategy combo with a different `n_total` (e.g. an ad hoc
smaller smoke-test run) silently loaded a stale, wrong-length cached file instead of
re-extracting -- crashing downstream with a sklearn "inconsistent numbers of samples"
error at best, or (same length, different seed) silently misaligning texts and labels
with no error at all at worst. See NOTES.md.

Uses pytest's tmp_path/monkeypatch throughout so this never touches the real repo's
results/features/ directory.
"""
import pytest

import aitext.pipeline as pipeline


@pytest.fixture(autouse=True)
def _isolated_features_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_FEATURES_DIR", tmp_path)


def test_feature_cache_path_differs_by_n_total():
    small_run = pipeline._feature_cache_path("mage", "bert", "score", n_total=40, seed=41)
    full_run = pipeline._feature_cache_path("mage", "bert", "score", n_total=10000, seed=41)

    assert small_run != full_run


def test_feature_cache_path_differs_by_seed():
    seed_a = pipeline._feature_cache_path("mage", "bert", "score", n_total=10000, seed=1)
    seed_b = pipeline._feature_cache_path("mage", "bert", "score", n_total=10000, seed=2)

    assert seed_a != seed_b


def test_feature_cache_path_uses_npy_only_for_embedding_strategy():
    assert pipeline._feature_cache_path("mage", "bert", "embedding", 10000, 41).suffix == ".npy"
    assert pipeline._feature_cache_path("mage", "bert", "score", 10000, 41).suffix == ".csv"
    assert pipeline._feature_cache_path("mage", "bert", "pawn", 10000, 41).suffix == ".csv"
    assert pipeline._feature_cache_path("mage", "bert", "zero_shot", 10000, 41).suffix == ".csv"
