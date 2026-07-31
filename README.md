# EVALUATING-DIFFUSION-LANGUAGE-MODELS-FOR-AI-GENERATED-TEXT-DETECTION

Support code for the paper *"Evaluating Diffusion Language Models for AI-Generated
Text Detection: A Multi-Strategy, Multi-Dataset Analysis"*. Evaluates autoregressive
models (GPT-2, LLaMA-2-7B, Llama-3.1-8B, a GPT-3 proxy), MLM models (BERT, RoBERTa,
ModernBERT), and diffusion models (LLaDA-8B-Base, LLaDA-8B-VRPO, Dream-LLaDA-7B) as
detection backbones under four strategies: global log-likelihood score, PAWN-style
token-level metrics, embedding classification, and zero-shot classification via a
fixed prompt. Datasets: MAGE, RAID, DeepfakeTextDetect (ATDP), Beemo, and M4GT-Bench.

## Structure

```
configs/
  models.yaml                 # model registry: paradigm, HF checkpoint, hyperparameters
  datasets.yaml                # dataset registry
  experiments/*.yaml            # what to run: dataset(s) + models + strategies + classifiers

src/aitext/
  models/        # ModelWrapper per paradigm (autoregressive / masked_lm / diffusion)
  datasets/       # loaders (mage / raid / deepfake_text_detect / beemo / m4gt_bench) + 50/50 balanced sampling
  metrics/pawn.py  # the 5 PAWN metrics + the sequence score, in one place
  metrics/zero_shot.py  # zero-shot prompt mechanics (Strategy 4), shared across the 3 paradigms
  classifiers/    # LogReg / RandomForest / XGBoost / DeepMLP + registry
  strategies/      # Score / PAWN / Embeddings / Zero-shot -- each calls model.<method>(texts)
  pipeline.py       # orchestrates dataset -> model -> strategy -> classifier, with caching

scripts/
  run_experiment.py         # runs one configs/experiments/*.yaml end-to-end
  build_paper_tables.py      # aggregates results/tables/*.csv into the paper's tables
  tune_diffusion_masking.py  # grid search over mask_ratio/num_samples for a diffusion model

results/
  features/   # cache of extracted scores/PAWN/embeddings (per dataset/model/strategy)
  tables/     # classification + performance results, ready for the paper's tables

tests/        # unit tests, no GPU/network needed (PAWN formulas, balancing, DeepMLP, scoring-bug regression)
legacy/notebooks/  # the original notebooks, kept as historical reference
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .   # makes `aitext` importable as a package (uses the src/ layout)
```

`raid` (the benchmark, used by `src/aitext/datasets/raid.py`) isn't under a trivial
PyPI name -- install it following https://github.com/liamdugan/raid.

`llama3_1_8b` (`meta-llama/Llama-3.1-8B`) is a **gated** repo on HuggingFace: accept
Meta's license on the model page and authenticate the machine before running anything
(`huggingface-cli login`, or the `HF_TOKEN` environment variable) -- otherwise
`from_pretrained` fails for that model specifically.

## Usage

Run an experiment (downloads models from HuggingFace, needs a GPU for the larger
models):

```bash
python scripts/run_experiment.py configs/experiments/mage.yaml
python scripts/run_experiment.py configs/experiments/raid.yaml
python scripts/run_experiment.py configs/experiments/deepfaketextdetect.yaml
python scripts/run_experiment.py configs/experiments/beemo.yaml
python scripts/run_experiment.py configs/experiments/m4gt_bench.yaml
python scripts/run_experiment.py configs/experiments/llada_family.yaml
```

`n_total` in `beemo.yaml`/`m4gt_bench.yaml` is a provisional value -- if
`balanced_sample` raises complaining about fewer available rows, adjust it to
whatever maximum the error reports.

Each run caches the extracted features (scores/PAWN/embeddings) in
`results/features/<dataset>/<model>_<strategy>.{csv,npy}`, so re-running the same
experiment after adding a new classifier doesn't repeat GPU inference.

Once you've run the experiments you care about, generate the paper's tables:

```bash
python scripts/build_paper_tables.py
```

## Adding something new

- **A model** (of an already-supported paradigm): one entry in `configs/models.yaml`.
  No code changes needed.
- **A dataset**: a loader in `src/aitext/datasets/<name>.py` that returns a
  DataFrame `["text", "label"]` (using `aitext.datasets.base.balanced_sample`), one
  line in `_LOADER_MODULES` in `src/aitext/datasets/registry.py`, and one entry in
  `configs/datasets.yaml`.
- **A classifier**: a `factory(seed) -> object with .fit/.predict_proba` function in
  `src/aitext/classifiers/`, registered in `_FACTORY_PATHS` in
  `src/aitext/classifiers/registry.py`.
- **A strategy**: a module in `src/aitext/strategies/` with `extract_features` and
  `evaluate`, registered in `STRATEGY_MODULES` in `src/aitext/strategies/__init__.py`.
  Not every strategy needs to train a classifier -- `strategies/zero_shot.py` is the
  example: `evaluate()` ignores the classifier list and evaluates the model's score
  directly against 100% of the labels via
  `aitext.classifiers.evaluation.direct_score_eval`, instead of `holdout_eval`/
  `cross_validated_eval`.
- **A genuinely new model paradigm**: a class implementing `ModelWrapper`
  (`src/aitext/models/base.py`) and a new branch in `_wrapper_class_for` in
  `src/aitext/models/registry.py`.

## Tests

```bash
pytest
```

Tests are purely unit-level (PAWN formulas, class balancing, DeepMLP shape, the
zero-shot prompt mechanics, the Beemo melt, the M4GT-Bench label flip, and a
regression test for the fixed scoring bug); they need no GPU, no downloads, and not
even the heavier dependencies (`transformers`/`datasets`/`xgboost` -- that's why each
loader's shared logic lives in `aitext.datasets.base`, not in the loader itself).
End-to-end validation of the full pipeline happens on the GPU machine.
