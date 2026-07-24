"""Model registry: reads configs/models.yaml and instantiates the right ModelWrapper
subclass for a given model name.

The paradigm classes are imported lazily (inside `_wrapper_class_for`), not at the
top of this file -- otherwise `import aitext.models` would require every paradigm's
own dependency (bitsandbytes for autoregressive.py/diffusion.py, transformers for all
three, ...) to be installed just to look up a model name. Adding a model of an
already-supported paradigm never touches this file -- just add an entry to
configs/models.yaml. A genuinely new paradigm needs one new branch here.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from aitext.models.base import ModelWrapper

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"


def _load_registry() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def list_models() -> list[str]:
    return list(_load_registry().keys())


def _wrapper_class_for(paradigm: str):
    if paradigm == "autoregressive":
        from aitext.models.autoregressive import AutoregressiveModel

        return AutoregressiveModel
    if paradigm == "masked_lm":
        from aitext.models.masked_lm import MaskedLMModel

        return MaskedLMModel
    if paradigm == "diffusion":
        from aitext.models.diffusion import DiffusionModel

        return DiffusionModel
    raise ValueError(f"Unknown paradigm '{paradigm}'")


def load_model(
    name: str,
    device: str = "cuda",
    max_length: int = 512,
    batch_size: int = 4,
) -> ModelWrapper:
    registry = _load_registry()
    if name not in registry:
        raise KeyError(f"Model '{name}' is not registered in {_CONFIG_PATH}")
    entry = registry[name]
    paradigm = entry["paradigm"]
    cls = _wrapper_class_for(paradigm)

    kwargs = dict(
        name=name,
        hf_id=entry["hf_id"],
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    if paradigm in ("autoregressive", "diffusion"):
        kwargs["quantize_4bit"] = entry.get("quantize_4bit", False)
    if paradigm == "diffusion":
        pawn_cfg = entry.get("pawn", {})
        kwargs["mask_ratio"] = pawn_cfg.get("mask_ratio", 0.35)
        kwargs["num_samples"] = pawn_cfg.get("num_samples", 10)
        kwargs["accepts_2d_attention_mask"] = entry.get("accepts_2d_attention_mask", True)

    return cls(**kwargs)
