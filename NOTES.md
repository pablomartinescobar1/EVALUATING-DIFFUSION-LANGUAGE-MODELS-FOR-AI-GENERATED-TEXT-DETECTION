# Discrepancias paper-vs-código encontradas durante la reestructuración

Este documento recoge lo encontrado al comparar los notebooks originales (ahora en
`legacy/notebooks/`) contra el texto del paper, separado en dos grupos: lo que este
refactor **ya corrige en el código** (así que los números de las tablas cambiarán la
próxima vez que corras `scripts/run_experiment.py`) y lo que **solo queda documentado**
porque cambiarlo es una decisión de investigación, no de ingeniería, y te corresponde
a ti decidir si lo abordas y cómo reflejarlo en el texto.

## Corregido en el código (recalcular antes de la próxima versión del paper)

1. **Bug de scoring autorregresivo.** `batch_autoregressive_scores` (GPT-2, LLaMA,
   GPT-3-proxy) leía `outputs.loss` de Hugging Face -- ya promediado sobre *todo el
   batch* -- y hacía `.repeat(len(batch))`, así que los 4 textos de un mismo batch
   recibían el mismo score. Coincide con que sus AUC de "Score" en las Tablas 2-4
   estaban pegados a 0.50. Corregido en
   [`src/aitext/models/autoregressive.py`](src/aitext/models/autoregressive.py) vía
   `aitext.metrics.pawn.sequence_log_likelihood` (cálculo per-ejemplo con
   `reduction="none"`, igual que ya hacían MLM/difusión). **Impacto esperado**: las
   filas "Score" de GPT-2/LLaMA/GPT-3 en las Tablas 2-4 probablemente suban de ~0.50 a
   algo más informativo -- re-ejecuta antes de dar por buena la afirmación "global
   scores fail regardless of paradigm".

2. **GPT-3 ausente de Embeddings.** En `FinalResultsMAGE.ipynb` el modelo GPT-3-proxy
   se cargaba pero no entraba en el loop de extracción de embeddings, pese a que la
   Tabla 2 reporta un valor para "GPT-3 Embed." Ahora cada estrategia
   (`src/aitext/strategies/embedding.py`) corre sobre *todos* los modelos del
   experimento por construcción -- ya no hay listas ad hoc que se puedan quedar cortas.

3. **MAGE no se balanceaba 5.000/5.000 explícitamente**, a diferencia de RAID y
   DeepfakeTextDetect que sí estratificaban. Ahora los tres datasets usan el mismo
   `aitext.datasets.base.balanced_sample`.

4. **Cobertura incompleta de la Tabla 5.** `FinalResultsLLADAs.ipynb` solo corría
   sobre MAGE con 1.000 muestras. `configs/experiments/llada_family.yaml` ahora cubre
   MAGE + RAID + DeepfakeTextDetect a 10.000 muestras, igual que los experimentos
   principales.

## Solo documentado -- requiere tu decisión

5. **`GSAI-ML/LLaDA-1.5` no es un modelo de 1.5B parámetros.** Confirmado en su ficha
   de Hugging Face: es LLaDA 1.5, un refinamiento por VRPO de **LLaDA-8B-Instruct**
   (8B parámetros reales). Decidiste mantener el experimento (ver
   `configs/models.yaml`, entrada `llada_8b_vrpo`), pero esto significa que **no existe
   ningún checkpoint LLaDA de escala pequeña** en la comparación actual. La Sección
   5.2 y la Tabla 5 del paper deben reescribirse: ya no es una comparación
   "small-scale vs. large-scale" de difusión, son dos variantes de 8B (Base vs. VRPO)
   frente a Dream-LLaDA-7B. Si quieres conservar la comparación "parameter-controlled"
   tal y como está planteada en la introducción (contribución 3), necesitarías un
   checkpoint LLaDA genuinamente pequeño (si existe alguno publicado) o replantear esa
   contribución.

6. **"GPT-3-medium" es en realidad `EleutherAI/gpt-neo-2.7B`**, un proxy abierto, no
   GPT-3 real de OpenAI (la API no expone log-probs de vocabulario completo para
   Score/PAWN). Es una práctica habitual pero debe declararse explícitamente en el
   paper -- ahora mismo el texto y la Tabla 1 lo presentan como si fuera GPT-3.

7. **"LLaMA-7B" es en realidad `NousResearch/Llama-2-7b-hf`** (LLaMA-**2**, no
   LLaMA-1). Precisar la versión en la Tabla 1 y el texto.

8. **El score de difusión (Estrategia 1) no es el ELBO variacional que describe la
   Sección 3.1.** El texto dice *"we use the model's variational lower bound on
   log-likelihood"*, pero el código (`DiffusionModel.score` en
   [`src/aitext/models/diffusion.py`](src/aitext/models/diffusion.py)) simplemente
   desplaza los logits una posición y calcula log-verosimilitud como si LLaDA fuera
   autorregresivo puro -- igual que en los notebooks originales. Se ha mantenido tal
   cual (no es un bug de refactorización, es una simplificación metodológica
   preexistente) porque "arreglarlo" cambiaría los números reportados; es una decisión
   de investigación, no de este refactor.

9. **El masking Monte Carlo (Estrategia 2, difusión) no usa "t ~ U[0,1]" como dice el
   texto.** Usa un `mask_ratio` fijo, calibrado por grid search (ver
   `scripts/tune_diffusion_masking.py`, que sustituye a los notebooks
   `LogisticRegresion+PawnMetrics.ipynb` / `MetricasFinales.ipynb` /
   `MorePatience.ipynb`), repetido `num_samples` veces con posiciones aleatorias
   distintas -- no un ratio nuevo muestreado de U[0,1] en cada pasada. `mask_ratio` y
   `num_samples` son ahora configurables por modelo en `configs/models.yaml` si decides
   implementar el muestreo aleatorio real más adelante. Mientras tanto, ajusta el texto
   de la Sección 3.2 para describir lo que de verdad se hizo.
