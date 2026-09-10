"""Verifies aitext.pipeline._n_total_for: per-dataset n_total override, used so one
experiment YAML can mix datasets whose natural pool is too small for the default
n_total (e.g. Beemo) with the rest, via an optional `n_total_overrides` map -- see
configs/experiments/qwen3_8b.yaml."""
from aitext.pipeline import _n_total_for


def test_uses_top_level_n_total_when_no_override_present():
    config = {"n_total": 10000}
    assert _n_total_for("mage", config) == 10000


def test_uses_override_for_the_matching_dataset_only():
    config = {"n_total": 10000, "n_total_overrides": {"beemo": 4374}}
    assert _n_total_for("beemo", config) == 4374
    assert _n_total_for("mage", config) == 10000


def test_missing_overrides_key_behaves_like_no_overrides():
    config = {"n_total": 5000}
    assert _n_total_for("raid", config) == 5000
