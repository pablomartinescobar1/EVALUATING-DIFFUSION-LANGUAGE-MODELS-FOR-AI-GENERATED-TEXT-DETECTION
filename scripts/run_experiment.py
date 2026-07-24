#!/usr/bin/env python
"""CLI entry point for a full experiment.

    python scripts/run_experiment.py configs/experiments/mage.yaml

Requires the `aitext` package to be importable -- either `pip install -e .` from the
repo root, or just run this script as-is (it adds src/ to sys.path as a fallback).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aitext.pipeline import load_experiment_config, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment config end-to-end.")
    parser.add_argument("config", help="Path to a configs/experiments/*.yaml file.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    print(f"Running experiment '{config.get('name', args.config)}'...")
    results = run_experiment(config)
    print(results.to_string(index=False))
    print(f"\nSaved to {config['output']}")


if __name__ == "__main__":
    main()
