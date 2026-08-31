# Asistente de Consulta de Derecho Laboral Individual

Asistente de apoyo para abogados junior colombianos: dado un texto de consulta, **identifica qué norma y qué artículo le aplican**. El sistema informa y verifica — nunca recomienda acciones sobre un caso concreto.

## Índice de archivos principales

| Archivo | Contenido |
|---|---|
| `notebooks/fine-tuning/fine_tuning_asistente_legal.ipynb` | **Entrega M1**: carga de BETO, LoRA, entrenamiento, baselines, evaluación y ejemplos cualitativos (corre end-to-end en Colab T4) |
| `notebooks/evaluation/harness_de_evaluacion.ipynb` | **Entrega M2**: harness de evaluación (métrica automática, LLM-as-a-judge, aciertos de dominio) y scorecard del sistema real de M1 |
| `s04_lora_adapter/` | Adaptador LoRA entrenado (pesos, no el backbone de BETO) usado por el harness de M2 |
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

El backbone de BETO queda **congelado**; se inyectan ramas de bajo rango (`W' = W + (α/r)·B·A`) en las capas de atención. Se entrenan las ramas LoRA y, vía `modules_to_save`, tanto la cabeza clasificadora como el `pooler` de BERT (`["classifier", "pooler"]`): **887 042 de 110 739 460 parámetros — 0.80%**. El `pooler` (Dense+Tanh sobre el token `[CLS]`) es lo que la cabeza clasificadora realmente lee — si se deja fuera de `modules_to_save`, PEFT nunca lo guarda y cada recarga del adaptador lo reinicializa al azar, desincronizado del classifier ya entrenado (ranking indeterminista entre cargas, verificado con una celda de reproducibilidad al final del notebook). Aun así, el fine-tuning cabe en una sesión Colab chica y reduce el riesgo de overfitting frente al full fine-tuning.

### Hiperparámetros principales

- **`r = 8`.** Corpus pequeño (749 pares): un rango bajo evita que la rama tenga capacidad de memorizar en vez de generalizar.
- **`lora_alpha = 16`.** Con `α/r = 2` la adaptación entra con fuerza suficiente para mover el modelo sin descarrilar las representaciones pre-entrenadas.
- **`target_modules = ["query", "value"]`.** Las proyecciones que deciden qué mirar de la consulta y de la norma; adaptar solo ahí basta y evita cuadruplicar los entrenables tocando MLP o la cabeza.
- **`warmup_steps`** (calculado, ~10% de los pasos totales, no un número fijo): se deriva de `len(train)` para no quedar desactualizado si el dataset crece, y evita el warning de `warmup_ratio` deprecado en transformers 5.x.

El resto de hiperparámetros (`lora_dropout`, `weight_decay`, batch/grad-accum, `num_train_epochs`, `fp16`, semilla) sigue configuración estándar de PEFT/Trainer, documentada como comentarios en el notebook.

### Early stopping

Mejor checkpoint por **F1 en validation** (`load_best_model_at_end`), con `patience=2`. Se entrenaron las **8 épocas configuradas** (el F1 siguió mejorando lo suficiente como para que `patience` nunca acumulara 2 caídas seguidas); el máximo (0.667) se dio en la **época 7**:

| Época | Loss val | Acc | Prec | Recall | F1 |
|---|---|---|---|---|---|
| 1 | 0.516 | 0.764 | 0.600 | 0.563 | 0.581 |
| 2 | 0.529 | 0.782 | 0.722 | 0.406 | 0.520 |
| 3 | 0.463 | 0.782 | 0.643 | 0.563 | 0.600 |
| 4 | 0.457 | 0.782 | 0.618 | 0.656 | 0.636 |
| 5 | 0.437 | 0.800 | 0.679 | 0.594 | 0.633 |
| 6 | 0.445 | 0.809 | 0.704 | 0.594 | 0.644 |
| 7 | 0.434 | 0.818 | 0.714 | 0.625 | **0.667** |
| 8 | 0.441 | 0.809 | 0.720 | 0.563 | 0.632 |

### Resultados (validation)

| Modelo | Acc | Prec | Recall | F1 |
|---|---|---|---|---|
| Clase mayoritaria | 0.709 | 0.000 | 0.000 | 0.000 |
| BETO zero-shot | 0.709 | 0.000 | 0.000 | 0.000 |
| **BETO + LoRA** | **0.818** | **0.714** | **0.625** | **0.667** |

Comparación oficial (fila por fila, mismo `validation`). El producto funciona como recuperación — para cada consulta se ordenan los candidatos y se muestran los top-k — así que la métrica principal es el ranking por sentencia: **Recall@3 0.912** vs. 0.544 de BETO zero-shot; **Precision@3 0.333** vs. 0.205 (clase mayoritaria es N/A: mismo score para todos no define un orden).

**Confirmación en `test`** (held-out, nunca visto durante entrenamiento ni selección de checkpoint — no reemplaza la comparación anterior, solo confirma que la mejora se sostiene):

| Modelo | Acc | Prec | Recall | F1 |
|---|---|---|---|---|
| Clase mayoritaria | 0.617 | 0.000 | 0.000 | 0.000 |
| BETO zero-shot | 0.617 | 0.000 | 0.000 | 0.000 |
| **BETO + LoRA** | **0.775** | **0.744** | **0.630** | **0.682** |

**Recall@3 0.898** vs. 0.478 de BETO zero-shot; **Precision@3 0.487** vs. 0.256.

**Lectura honesta.** Estos números bajan frente a una versión anterior sin los negativos de la Sección 1.1 (F1 val 0.76, test 0.88) — es una compensación deliberada, no una regresión: esa versión anterior rankeaba bien dentro de pools chicos de 3-5 candidatos por sentencia, pero el producto rankea contra el diccionario completo (143 artículos), donde colapsaba en un puñado de artículos "genéricos" sin importar la consulta (verificado en `harness_de_evaluacion.ipynb`). Los negativos adicionales cuestan precision/recall en la tarea de clasificación fila-por-fila, pero el ranking por sentencia mejora (**Recall@3 0.91 val, 0.90 test**, ambos por encima de la meta de 0.85) y el sistema completo, evaluado extremo a extremo contra el harness de M2, mejora de **0/10 a 5/10** en aciertos de dominio frente a la versión sin estos negativos (detalle e interpretación honesta en la sección 8 de `harness_de_evaluacion.ipynb` — buena parte de esos aciertos todavía pasan por similitud de embeddings, no por precisión real del juez). Precision 0.74 en test sigue confirmando que lo que el modelo marca como relacionado casi nunca es ruido.

### Ejemplos cualitativos

```
1. Verdadero positivo representativo (SL-2338/23)
Consulta : "Trabajé como pintor en Industrias Mussgo desde septiembre de 2000 hasta
            diciembre de 2016, durante 16 años. Me decían que era contratista independiente
            y me pagaban con cuentas de cobro, pero yo trabajaba en sus instalaciones..."
Artículo : Ley 100 de 1993, Art. 17 (obligatoriedad de cotizaciones)
Esperado : RELEVANTE   →   Predicho: RELEVANTE (score 0.798)  ✅ acierta

2. Falso negativo más cercano al umbral (SL-2338/23, misma consulta)
Artículo : CST Art. 23 (elementos esenciales del contrato de trabajo)
Esperado : RELEVANTE   →   Predicho: NO RELEVANTE (score 0.367)  ❌ falla

3. Negativo difícil correctamente rechazado (T-014/26)
Consulta : "Tres mujeres fueron desvinculadas de sus trabajos: una cuando estaba lactando
            (su bebé tenía 6 meses), otra desvinculada dos veces (embarazada primero, y
            luego cuando su hija tenía 4 meses)..."
Artículo : CST Art. 240 (permiso para despedir a trabajadora embarazada)
Esperado : NO RELEVANTE   →   Predicho: NO RELEVANTE (score 0.476)  ✅ acierta

4. Falso positivo con mayor score (T-1097/08)
Consulta : "Trabajé como ayudante de municiones en varias ocasiones con contrato por obra.
            Me cayó un taladro en la pierna en agosto de 2007 y la empresa no reportó el
            accidente... me despidieron cuando terminó la obra, estando incapacitado..."
Artículo : Ley 6 de 1945 Art. 1 (definición de contrato de trabajo)
Esperado : NO RELEVANTE   →   Predicho: RELEVANTE (score 0.887)  ❌ falla
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
5. **Negativos adicionales** (sección 1.1 del notebook, sin inventar contenido legal): 106 de los 143 artículos del diccionario solo aparecían como positivo en el dataset (nunca como negativo) — sin contraste, el cross-encoder aprendía a puntuarlos alto sin importar la consulta al rankear contra el corpus completo (el régimen real del producto, no el de entrenamiento original de pools chicos). Se agregan (a) cada artículo "solo positivo" emparejado con una consulta real que no lo necesita, y (b) 35 consultas genéricas sin nada que ver con derecho laboral, para que el modelo aprenda a devolver scores bajos ante texto fuera de dominio en vez de citar cualquier artículo con confianza → **749 utilizables**.
6. **Split por sentencia** (`GroupShuffleSplit`, seed 42): evita que positivo y negativos de una misma consulta caigan en splits distintos (fuga); las 35 consultas fuera de dominio cuentan como sentencias propias, sin positivos.
7. **Tokenización**: 512 tokens, `truncation="only_second"` — la consulta nunca se recorta.

### Tamaño, split e idioma

561 pares reales (285 positivos, 138 negativos fáciles, 138 negativos difíciles) → 538 utilizables + 211 negativos adicionales (106 aleatorios de corpus completo + 105 fuera de dominio) → **749 utilizables** (label 0: 487 / 1: 262):

| Partición | Pares | Sentencias | Positivos |
|---|---|---|---|
| Train | 519 | 121 | 184 |
| Validation | 110 | 26 | 32 |
| Test | 120 | 26 | 46 |

**Idioma:** español. **Licencia:** normas y sentencias son de dominio público (fuentes oficiales del Estado colombiano); los pares de entrenamiento (extracción y negativos vía LLM) son material propio del equipo. Código: MIT.

**Sesgos y limitaciones.** El corpus depende de qué llega a instancias judiciales superiores: sobrerrepresenta estabilidad laboral reforzada (embarazo, salud, discapacidad — lo que más llega a tutela) y subrepresenta jornada/recargos y factores salariales; vacaciones está casi en cero pese a haberse probado varios canales de búsqueda, posible hueco estructural de la fuente, no de cobertura.

## Reproducibilidad y equipo

`notebooks/fine-tuning/fine_tuning_asistente_legal.ipynb` corre de punta a punta en Colab con seed 42 fija y outputs preservados. `REF` de datos apunta a `main`. Las claves API solo se usan en la recolección del dataset (`notebooks/dataset/`).

**Equipo — "Lawten"**: Pablo Cabrejos, Miguel Ángel Ortiz, Martín Valencia, Samuel López.
