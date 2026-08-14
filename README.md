# Asistente de Consulta de Derecho Laboral Individual

Asistente de apoyo para abogados junior colombianos: dado un texto de consulta, **identifica qué norma y qué artículo le aplican**. El sistema informa y verifica — nunca recomienda acciones sobre un caso concreto.

## Índice de archivos principales

| Archivo | Contenido |
|---|---|
| `notebooks/finetuning/Fine_tuning_Asistente_legal.ipynb` | **Entrega M1**: carga de BETO, LoRA, entrenamiento, baselines, evaluación y ejemplos cualitativos (corre end-to-end en Colab T4) |
| `notebooks/decisiones_base_encoder.ipynb` | Justificación de familia/modelo base (tokenización de dominio) |
| `notebooks/dataset/` | Recolección y pre-procesamiento del corpus: `ds_parte1_redal`, `ds_parte2_corte_const`, `ds_parte3_research_list`, `ds_parte4_sl_casacion`, `ds_diccionario_articulos` |
| `data/` | Dataset de entrenamiento e insumos: `dataset_cross_encoder.csv`, `diccionario_articulos.csv`, `fuentes_normas.csv`, `descartados.csv`, `candidatas_research.csv` |
| `docs/` | Definición del proyecto (`STAI_Plantilla_Proyecto.md`) |

## Reporte M1 — Fine-tuning con LoRA (BETO)

### Tarea y métricas

**Tarea.** Clasificación binaria de pares (cross-encoder): `[CLS] consulta [SEP] norma + artículo [SEP]` → softmax sobre los logits da `P(relacionado)`, la **score de relevancia** que ordena los candidatos en el producto.

**Métricas.** Para la clasificación binaria se usa **F1** (no accuracy: las clases están desbalanceadas y el falso negativo de una norma aplicable es el error más costoso). Como métrica de producto, **Recall@3 y Precision@3**: de los 3 artículos con mayor score por consulta, ¿qué fracción de los realmente aplicables se recuperó y qué fracción es correcta? Meta: **Recall@3 ≥ 0.85**.

### Modelo base: por qué BETO

`dccuchile/bert-base-spanish-wwm-cased` (BETO), BERT monolingüe en español con *Whole Word Masking*:

1. **Pre-entrenamiento ↔ tarea.** BETO usa MLM (predice palabras enmascaradas por contexto), lo que entrena representaciones sensibles al léxico jurídico español — la base que nuestra tarea discriminativa afina.
2. **Tokenización de dominio.** Fertilidad 1.35 tokens/palabra sobre texto jurídico-laboral vs. 1.56 de mBERT/DistilBERT (~14% menos tokens). DistilBERT además usa una arquitectura de atención (`q_lin`/`v_lin`) distinta al patrón LoRA estándar de BETO.
3. **Tamaño.** ~110M de parámetros, corre holgado en el T4 gratis de Colab.

Justificación completa en `notebooks/decisiones_base_encoder.ipynb`.

### Baselines

Dos referencias, evaluadas sobre el mismo `validation`: **clase mayoritaria** (el piso, sin leer texto) y **BETO sin afinar (zero-shot)** — el baseline que recomienda la asignación, porque usa el mismo encoder del modelo final y aísla exactamente el aporte del fine-tuning.

### LoRA: qué se congela y qué se entrena

El backbone de BETO queda **congelado**; se inyectan ramas de bajo rango (`W' = W + (α/r)·B·A`) en las capas de atención. Solo se entrenan las ramas LoRA y la cabeza clasificadora nueva (vía `modules_to_save`): **296 450 de 110 148 868 parámetros — 0.27%**. Esto hace que el fine-tuning quepa en una sesión Colab chica, sea estable con solo 538 pares y reduzca el riesgo de overfitting frente al full fine-tuning.

### Hiperparámetros principales

- **`r = 8`.** Corpus pequeño (538 pares): un rango bajo evita que la rama tenga capacidad de memorizar en vez de generalizar.
- **`lora_alpha = 16`.** Con `α/r = 2` la adaptación entra con fuerza suficiente para mover el modelo sin descarrilar las representaciones pre-entrenadas.
- **`target_modules = ["query", "value"]`.** Las proyecciones que deciden qué mirar de la consulta y de la norma; adaptar solo ahí basta y evita cuadruplicar los entrenables tocando MLP o la cabeza.
- **`warmup_steps`** (calculado, ~10% de los pasos totales, no un número fijo): se deriva de `len(train)` para no quedar desactualizado si el dataset crece, y evita el warning de `warmup_ratio` deprecado en transformers 5.x.

El resto de hiperparámetros (`lora_dropout`, `weight_decay`, batch/grad-accum, `num_train_epochs`, `fp16`, semilla) sigue configuración estándar de PEFT/Trainer, documentada como comentarios en el notebook.

### Early stopping

Mejor checkpoint por **F1 en validation** (`load_best_model_at_end`), con `patience=2`. De 8 épocas configuradas se entrenaron **6**; el F1 alcanzó su máximo (0.763) en la **época 4** y se estabilizó:

| Época | Loss val | Acc | Prec | Recall | F1 |
|---|---|---|---|---|---|
| 1 | 0.702 | 0.540 | 0.778 | 0.156 | 0.259 |
| 2 | 0.644 | 0.678 | 0.905 | 0.422 | 0.576 |
| 3 | 0.543 | 0.782 | 0.933 | 0.622 | 0.747 |
| 4 | 0.492 | 0.793 | 0.935 | 0.644 | **0.763** |
| 5 | 0.452 | 0.793 | 0.935 | 0.644 | 0.763 |
| 6 | 0.465 | 0.793 | 0.935 | 0.644 | 0.763 |

### Resultados (validation)

| Modelo | Acc | Prec | Recall | F1 |
|---|---|---|---|---|
| Clase mayoritaria | 0.483 | 0.000 | 0.000 | 0.000 |
| BETO zero-shot | 0.483 | 0.000 | 0.000 | 0.000 |
| **BETO + LoRA** | **0.793** | **0.935** | **0.644** | **0.763** |

Comparación oficial (fila por fila, mismo `validation`). El producto funciona como recuperación — para cada consulta se ordenan los candidatos y se muestran los top-k — así que la métrica principal es el ranking por sentencia: **Recall@3 0.898** vs. 0.679 de BETO zero-shot; **Precision@3 0.587** vs. 0.381 (clase mayoritaria es N/A: mismo score para todos no define un orden).

**Confirmación en `test`** (held-out, nunca visto durante entrenamiento ni selección de checkpoint — no reemplaza la comparación anterior, solo confirma que la mejora se sostiene):

| Modelo | Acc | Prec | Recall | F1 |
|---|---|---|---|---|
| Clase mayoritaria | 0.553 | 0.000 | 0.000 | 0.000 |
| BETO zero-shot | 0.553 | 0.000 | 0.000 | 0.000 |
| **BETO + LoRA** | **0.895** | **0.906** | **0.853** | **0.879** |

**Lectura honesta.** F1 0.76: precision 0.94 confirma que lo que el modelo marca como relacionado casi nunca es ruido, pero recall 0.64 muestra que todavía se le escapan normas aplicables — el asistente prefiere no afirmar antes que afirmar mal, razonable para su rol de verificación. El delta frente al baseline (0 → 0.76 F1) es todo aporte del fine-tuning. Meta del proyecto (Recall@3 ≥ 0.85) **cumplida**.

### Ejemplos cualitativos

```
Consulta : "Trabajé 24 años como operario de montacargas y me despidieron después de que
            un compañero ebrio me agredió verbalmente con insultos racistas... yo
            reaccioné lanzándole mi casco..."
Artículo : Ley 50 de 1990, Art. 6 (sustitución de personal por incapacidad/licencia)
Esperado : RELEVANTE   →   Predicho: RELEVANTE (score 0.795)  ✅

Consulta : "Me despidieron de Helicol después de 9 años trabajando, sin saber que estaba
            embarazada... la EPS me negó la licencia de maternidad..."
Artículo : CST Art. 239 (protección a la maternidad)
Esperado : NO RELEVANTE   →   Predicho: RELEVANTE (score 0.665)  ❌ falso positivo
```

Selección reproducible sobre `validation` (verdadero positivo representativo, falso negativo cercano al umbral, negativo difícil bien rechazado, falso positivo con mayor score) en la sección 8 del notebook.

## Dataset

Dos fases: **recolección** (sentencias → pares pregunta-norma, en `notebooks/dataset/`) y **adecuación** (pares → texto listo para tokenizar, al inicio del notebook de fine-tuning), unidas por un join contra un diccionario de normas.

### Archivos

| Archivo | Contenido | Filas |
|---|---|---|
| `data/dataset_cross_encoder.csv` | Dataset final de pares: `consulta`, `articulo`, `tipo`, `label`, `sentencia_origen` | 561 (138 sentencias) |
| `data/diccionario_articulos.csv` | Texto normativo completo, unido por `(fuente, numero)` | 143 artículos / 41 normas |
| `data/fuentes_normas.csv` | URL oficial resuelta a mano de cada norma | 41 |
| `data/descartados.csv` | Blacklist compartida de sentencias descartadas | 1362 |
| `data/candidatas_research.csv` | Candidatas de la vía de investigación externa | 130 |

### Pipeline

1. **Recolección** por 4 vías (`notebooks/dataset/`): cada sentencia pasa un filtro barato (Claude Haiku) antes de la extracción completa (Claude Sonnet), que produce la `consulta` en lenguaje coloquial y los artículos determinantes (`positivo`). Cada positivo suma un negativo fácil (tema distinto) y un negativo difícil (artículo real, cercano, que no aplica). Las 4 vías comparten `descartados.csv`.
2. **Diccionario de artículos**: texto íntegro de cada norma citada, extraído de portales oficiales vía `fuentes_normas.csv`.
3. **Join**: la cita se separa en `(fuente, numero)` normalizado y se une al diccionario. El texto no vive duplicado en cada fila del dataset.
4. **Limpieza**: 23 pares se descartan por no tener texto normativo con el que entrenar — 21 citan una ley/decreto completo sin número de artículo (nada específico que unir) y 2 citan convenciones colectivas privadas entre una empresa y un sindicato, no codificadas en ningún repositorio público → 538 utilizables.
5. **Split por sentencia** (`GroupShuffleSplit`, seed 42): evita que positivo y negativos de una misma consulta caigan en splits distintos (fuga).
6. **Tokenización**: 512 tokens, `truncation="only_second"` — la consulta nunca se recorta.

### Tamaño, split e idioma

561 pares (285 positivos, 138 negativos fáciles, 138 negativos difíciles) → **538 utilizables** (label 0: 276 / 1: 262):

| Partición | Pares | Sentencias | Positivos |
|---|---|---|---|
| Train | 375 | 96 | 183 |
| Validation | 87 | 21 | 45 |
| Test | 76 | 21 | 34 |

**Idioma:** español. **Licencia:** normas y sentencias son de dominio público (fuentes oficiales del Estado colombiano); los pares de entrenamiento (extracción y negativos vía LLM) son material propio del equipo. Código: MIT.

**Sesgos y limitaciones.** El corpus depende de qué llega a instancias judiciales superiores: sobrerrepresenta estabilidad laboral reforzada (embarazo, salud, discapacidad — lo que más llega a tutela) y subrepresenta jornada/recargos y factores salariales; vacaciones está casi en cero pese a haberse probado varios canales de búsqueda, posible hueco estructural de la fuente, no de cobertura.

## Reproducibilidad y equipo

`notebooks/finetuning/Fine_tuning_Asistente_legal.ipynb` corre de punta a punta en Colab con seed 42 fija y outputs preservados. `REF` de datos apunta a `main`. Las claves API solo se usan en la recolección del dataset (`notebooks/dataset/`).

**Equipo — "Lawten"**: Pablo Cabrejos, Miguel Ángel Ortiz, Martín Valencia, Samuel López.
