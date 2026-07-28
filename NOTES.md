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

## Ampliación post-revisión del tutor (Estrategia 4, datasets nuevos, modelos nuevos)

Tras la revisión del tutor sobre el borrador del paper, se añadió lo siguiente. Igual
que el resto de este documento, separo lo que ya está en el código de lo que sigue
siendo una decisión tuya para el texto del paper.

10. **Estrategia 4 (zero-shot) añadida.** `src/aitext/strategies/zero_shot.py` +
    `aitext.metrics.zero_shot` + `ModelWrapper.zero_shot_score` (una implementación
    por paradigma). Un prompt fijo y mínimo ("Is the text above written by a human
    (not an AI)? Answer (Yes/No):") compara `logP(" Yes") - logP(" No")` en una única
    posición -- para autorregresivos, la posición siguiente al prompt; para MLM/
    difusión, un `[MASK]` insertado como id ya resuelto (nunca como string, porque al
    menos un tokenizer de difusión no expone `mask_token` como string -- por eso
    `DiffusionModel` ya necesita su fallback `vocab_size - 1`). No se entrena ningún
    clasificador (`aitext.classifiers.evaluation.direct_score_eval` evalúa el score
    directamente contra el 100% de las etiquetas) -- eso es lo que la hace "zero-shot"
    frente a las Estrategias 1-3. El fine-tuning/LoRA que el tutor también sugirió
    como alternativa queda **fuera de alcance a propósito**: el paper enmarca
    deliberadamente la pregunta como "hasta dónde llegan los modelos de difusión SIN
    entrenar, ya sea como extractores de características (Estrategias 1-3) o
    zero-shot (Estrategia 4)".

11. **`llama7b` renombrado a `llama2_7b`** en `configs/models.yaml` y en los YAMLs de
    experimento -- mismo checkpoint (`NousResearch/Llama-2-7b-hf`, ver #7 arriba), solo
    la clave cambia, ahora que hay un segundo LLM (`llama3_1_8b`) y la versión importa
    más en las tablas. Sin riesgo: no había ninguna corrida cacheada en `results/`
    antes de este cambio.

12. **`llama3_1_8b` (`meta-llama/Llama-3.1-8B`) es un repo GATED en HuggingFace.**
    Hace falta que la cuenta HF del usuario haya aceptado la licencia de Meta y que la
    máquina de GPU esté autenticada (`huggingface-cli login` o `HF_TOKEN`) antes de
    `from_pretrained` -- no es un problema de código, es un prerequisito operativo.

13. **M4GT-Bench (`src/aitext/datasets/m4gt_bench.py`) usa un mirror comunitario, no
    el dataset oficial.** El repo oficial (`mbzuai-nlp/M4GT-Bench`) solo distribuye
    por Google Drive + `gdown`; se usa en su lugar `d0rj/SemEval2024-task8` (config
    `subtaskA_monolingual`) en HuggingFace `datasets`. Su convención de labels nativa
    es 0=humano/1=máquina -- **la opuesta** a la de este proyecto -- y el loader la
    invierte explícitamente (`aitext.datasets.base.invert_binary_label`).

14. **Alcance de M4GT-Bench reducido a solo inglés, decisión tuya ya tomada.** Se usa
    `subtaskA_monolingual`, no `subtaskA_multilingual` (9 idiomas), para mantener
    consistencia con el resto del pipeline -- tokenizers, el masking PAWN y el prompt
    zero-shot son todos de inglés. **Esto requiere un cambio en el texto del paper**:
    la Tabla 2 actualmente dice "Lang.=9" para M4GT-Bench; debe pasar a "Lang.=1", y
    conviene una nota en Limitations señalando que la cobertura multilingüe se dejó
    fuera a propósito (posible trabajo futuro).

15. **La Tabla 7 del borrador ("parameter-controlled": BERT/RoBERTa/ModernBERT
    "small-scale" vs. LLaDA-1.5B) no se puede construir tal y como está descrita.**
    Sigue sin existir ningún checkpoint LLaDA de escala pequeña (ver #5 arriba --
    reconfirmado durante esta ampliación, no ha cambiado). `scripts/build_paper_tables.py`
    no fabrica esa comparación; en su lugar genera dos tablas honestas por separado:
    una comparación de encoders pequeños (BERT/RoBERTa/ModernBERT entre sí, sin
    comparar contra difusión) y una comparación de modelos grandes (~7-8B, LLaMA-2-7B/
    Llama-3.1-8B frente a LLaDA-8B-Base/VRPO/Dream-LLaDA-7B -- esta sí es
    parámetro-comparable de verdad). El texto de la Sección 5.2 / Tabla 7 necesita
    reescribirse en consecuencia, igual que ya pedía el punto #5 para la Tabla 5
    original.

16. **`n_total` de Beemo y M4GT-Bench son valores provisionales**, no verificados
    contra el `value_counts()` real (esta sesión no tuvo acceso a red/GPU para
    cargarlos). `configs/experiments/beemo.yaml` usa 4374 (2.187/2.187, el tamaño
    íntegro del pool de Beemo tras el melt); `configs/experiments/m4gt_bench.yaml` usa
    10000 (igual escala que MAGE/RAID/ATDP; el split `train` tiene ~120k filas, así
    que debería sobrar margen). Si `balanced_sample` falla alto al lanzar el
    experimento real, ajusta `n_total` al máximo que indique el error.

17. **`configs/experiments/llada_family.yaml` sigue cubriendo solo
    `[mage, raid, deepfaketextdetect]`**, no se amplió a Beemo/M4GT-Bench -- el PAWN
    de difusión es caro (~6h/10k muestras por modelo, Tabla 9 del paper) y duplicar
    eso en 2 datasets más no se pidió explícitamente. Si se quiere, es un cambio de
    una línea (añadir a `datasets:`).
