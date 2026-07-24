# EVALUATING-DIFFUSION-LANGUAGE-MODELS-FOR-AI-GENERATED-TEXT-DETECTION

Código de soporte del paper *"Evaluating Diffusion Language Models for AI-Generated
Text Detection: A Multi-Strategy, Multi-Dataset Analysis"*. Evalúa modelos
autorregresivos (GPT-2, LLaMA-2-7B, un proxy de GPT-3), MLM (BERT, RoBERTa) y de
difusión (LLaDA-8B-Base, LLaDA-8B-VRPO, Dream-LLaDA-7B) como backbones de detección
bajo tres estrategias: score global de log-verosimilitud, métricas token-level tipo
PAWN, y clasificación sobre embeddings.

## Estructura

```
configs/
  models.yaml                 # registro de modelos: paradigma, checkpoint HF, hiperparámetros
  datasets.yaml                # registro de datasets
  experiments/*.yaml            # qué correr: dataset(s) + modelos + estrategias + clasificadores

src/aitext/
  models/        # ModelWrapper por paradigma (autoregressive / masked_lm / diffusion)
  datasets/       # loaders (mage / raid / deepfake_text_detect) + muestreo balanceado 50/50
  metrics/pawn.py  # las 5 métricas PAWN + el score de secuencia, en un único lugar
  classifiers/    # LogReg / RandomForest / XGBoost / DeepMLP + registro
  strategies/      # Score / PAWN / Embeddings -- cada una llama a model.<método>(texts)
  pipeline.py       # orquesta dataset -> modelo -> estrategia -> clasificador, con caché

scripts/
  run_experiment.py         # corre un configs/experiments/*.yaml de principio a fin
  build_paper_tables.py      # agrega results/tables/*.csv en las tablas del paper
  tune_diffusion_masking.py  # grid search de mask_ratio/num_samples para un modelo de difusión

results/
  features/   # caché de scores/PAWN/embeddings extraídos (por dataset/modelo/estrategia)
  tables/     # resultados de clasificación + rendimiento, listos para las tablas del paper

tests/        # unitarios, sin GPU ni red (fórmulas PAWN, balanceo, DeepMLP, regresión del bug de scoring)
legacy/notebooks/  # los notebooks originales, conservados como referencia histórica
NOTES.md            # discrepancias encontradas entre el código y el texto del paper
```

## Instalación

```bash
pip install -r requirements.txt
pip install -e .   # hace `aitext` importable como paquete (usa src/ layout)
```

`raid` (el benchmark, usado por `src/aitext/datasets/raid.py`) no está en un nombre
PyPI trivial -- instálalo siguiendo https://github.com/liamdugan/raid.

## Uso

Correr un experimento (descarga modelos de HuggingFace, requiere GPU para los modelos
grandes):

```bash
python scripts/run_experiment.py configs/experiments/mage.yaml
python scripts/run_experiment.py configs/experiments/raid.yaml
python scripts/run_experiment.py configs/experiments/deepfaketextdetect.yaml
python scripts/run_experiment.py configs/experiments/llada_family.yaml
```

Cada corrida cachea los features extraídos (scores/PAWN/embeddings) en
`results/features/<dataset>/<modelo>_<estrategia>.{csv,npy}`, así que volver a lanzar
el mismo experimento tras añadir un clasificador nuevo no repite la inferencia por GPU.

Una vez corridos los experimentos que te interesen, generar las tablas del paper:

```bash
python scripts/build_paper_tables.py
```

## Añadir algo nuevo

- **Un modelo** (de un paradigma ya soportado): una entrada en `configs/models.yaml`.
  No hace falta tocar código.
- **Un dataset**: un loader en `src/aitext/datasets/<nombre>.py` que devuelva un
  DataFrame `["text", "label"]` (usando `aitext.datasets.base.balanced_sample`), una
  línea en `_LOADER_MODULES` de `src/aitext/datasets/registry.py`, y una entrada en
  `configs/datasets.yaml`.
- **Un clasificador**: una función `factory(seed) -> objeto con .fit/.predict_proba`
  en `src/aitext/classifiers/`, registrada en `_FACTORY_PATHS` de
  `src/aitext/classifiers/registry.py`.
- **Una estrategia**: un módulo en `src/aitext/strategies/` con `extract_features` y
  `evaluate`, registrado en `STRATEGY_MODULES` de `src/aitext/strategies/__init__.py`.
- **Un paradigma de modelo genuinamente nuevo**: una clase que implemente
  `ModelWrapper` (`src/aitext/models/base.py`) y una rama nueva en
  `_wrapper_class_for` de `src/aitext/models/registry.py`.

## Tests

```bash
pytest
```

Los tests son puramente unitarios (lógica de PAWN, balanceo de clases, forma del
DeepMLP, y un test de regresión para el bug de scoring corregido -- ver
[NOTES.md](NOTES.md)); no requieren GPU ni descargar modelos. La validación end-to-end
del pipeline completo se hace en la máquina con GPU.

## Antes de la próxima versión del paper

Lee [NOTES.md](NOTES.md): recoge tanto los bugs que este refactor ya corrige (y que
por tanto cambiarán los números de las Tablas 2-4 al recalcular) como las
discrepancias que solo quedan documentadas porque su corrección es una decisión de
investigación tuya, no de ingeniería (el checkpoint "LLaDA-1.5B", el proxy de GPT-3,
la versión real de LLaMA, y la fidelidad de las Estrategias 1 y 2 de difusión respecto
a lo descrito en el texto).
