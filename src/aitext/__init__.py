"""AI-generated text detection pipeline: autoregressive, masked-LM, and diffusion
backbones evaluated as scoring instruments.

Deliberately import-light: submodules are meant to be imported directly
(`from aitext.pipeline import run_experiment`, `from aitext.metrics.pawn import ...`,
`from aitext.models.registry import load_model`) rather than re-exported here, so that
e.g. testing pure metric math never requires every dataset/model backend's own
dependency (transformers, bitsandbytes, the HF `datasets` package, `raid`, ...) to be
installed. See `aitext.datasets.registry` / `aitext.models.registry` for the same
lazy-import pattern applied to dataset loaders and model wrappers.
"""
