# Asistente de Consulta de Derecho Laboral Individual

Asistente de apoyo para abogados junior colombianos: dado un texto de consulta, **identifica qué norma y qué artículo le aplican**. El sistema informa y verifica — nunca recomienda acciones sobre un caso concreto.

## Índice de archivos principales

| Archivo | Contenido |
|---|---|
| `notebooks/finetuning/Fine_tuning_Asistente_legal.ipynb` | **Entrega M1**: carga de BETO, LoRA, entrenamiento, baselines, evaluación y ejemplos cualitativos (corre end-to-end en Colab T4) |
| `notebooks/decisiones_base_encoder.ipynb` | Justificación de familia/modelo base (tokenización de dominio) |
| `notebooks/dataset/` | Recolección y pre-procesamiento del corpus: `ds_parte1_redal`, `ds_parte2_corte_const`, `ds_parte3_research_list`, `ds_parte4_sl_casacion`, `ds_diccionario_articulos` |
| `data/` | Dataset de entrenamiento e insumos: `dataset_cross_encoder.csv`, `diccionario_articulos.csv`, `fuentes_normas.csv`, `descartados.csv`, `candidatas_research.csv` |
| `docs/` | Definición del proyecto y descripción de la asignación |

## Reporte M1 — Fine-tuning con LoRA (BETO)

### Tarea y métricas

**Tarea.** Clasificación binaria de pares (cross-encoder): `[CLS] consulta [SEP] norma + artículo [SEP]` → la cabeza clasificadora emite un par de **logits** (números reales sin normalizar, uno por clase). Aplicando softmax sobre ellos se obtiene `P(relacionado)`, la **score de relevancia** que ordena los candidatos en el producto. No es clasificación de un solo texto ni generación: procesa los dos textos juntos y produce un solo número por par.

**Métrica principal.** Para la clasificación binaria se usa **F1**, no accuracy: las clases están desbalanceadas en el espacio de recuperación real (muchos más candidatos no relevantes que relevantes) y el falso negativo de una norma aplicable es el error más costoso. Como métrica de producto (lo que ve el abogado) se usa **Recall@k y Precision@k** con k=3: para cada consulta se ordenan los artículos candidatos por score y se mira el top-3 — ¿qué fracción de los artículos realmente aplicables quedó en el top-k (recall) y qué fracción del top-k es correcta (precision)? Meta del proyecto: **Recall@3 ≥ 0.85**.

### Modelo base y familia: por qué BETO y no otro

Usamos `dccuchile/bert-base-spanish-wwm-cased` (**BETO**), un encoder BERT monolingüe en español con *Whole Word Masking*. Tres razones, en orden de peso:

1. **Conexión pre-entrenamiento ↔ tarea.** BETO fue pre-entrenado con **MLM (masked language modeling)**: predice palabras enmascaradas usando el contexto. Esto entrena representaciones sensibles a contexto y a léxico jurídico español. Nuestra tarea discriminativa (decidir si un artículo *responde* una consulta) afina esas representaciones — el objetivo de pre-entrenamiento alimenta directamente la tarea.
2. **Tokenización de dominio (consejo Lab A).** Comparando fertilidad (tokens por palabra) sobre texto jurídico-laboral español, BETO obtiene **1.35** frente a **1.56** de las alternativas multilingües mBERT/DistilBERT: ~**14% menos tokens** sobre el mismo texto. Un modelo que parte términos clave del dominio ("prestaciones", "cesantías", "indemnización") en muchas subpalabras rinde peor y es más costoso. Se descartó DistilBERT además por su arquitectura de atención (`q_lin`/`v_lin`), incompatible con el patrón LoRA estándar de BETO.
3. **Tamaño.** ~110M de parámetros, corre holgado en el T4 gratis de Colab con margen para adaptación eficiente.

Justificación completa en `notebooks/decisiones_base_encoder.ipynb`.

### Baselines (contra qué comparamos)

La asignación pide dos referencias. **Clase mayoritaria**: predice siempre la clase más frecuente en train (0, NO RELEVANTE: 192/375) sin leer el texto; comprueba cuánto se logra sin el modelo. **BETO sin afinar (zero-shot)**: el mismo encoder del modelo final pero con cabeza binaria recién inicializada y sin entrenamiento en esta tarea. Ambos se evalúan sobre el **mismo `validation`** para aislar el delta que aporta el fine-tuning.

### LoRA: qué se congela y qué se entrena

En **LoRA (Low-Rank Adaptation)** el peso original W del backbone queda **congelado** (no recibe gradientes) y se inyectan ramas paralelas de bajo rango: `W' = W + (α/r)·B·A`, donde `A` es una matriz aleatoria `d×r`, `B` es una matriz en ceros `r×d` (la rama parte en 0 y solo B se actualiza), `r` el rango y `α` la escala (factor efectivo `α/r = 2`).

En nuestro modelo: el **99.73% de los parámetros queda congelado** (backbone completo de 12 capas, embedding y capas de atención no objetivo). Se entrenan solo:

- **294 912 parámetros de LoRA**: 12 capas × 2 módulos (`query`, `value`) × 2 matrices (8×768 + 768×8) = 12 288 por módulo.
- **1 538 parámetros de la cabeza** (`classifier`, 768×2 + 2 bias), incluida vía `modules_to_save` porque la cabeza nueva no viene pre-entrenada.

**Total entrenable: 296 450 / 110 148 868 = 0.27 %.** Por eso el fine-tuning cabe en una sesión Colab pequeña, es estable con un corpus de 538 pares y reduce el riesgo de overfitting frente al full fine-tuning (110M parámetros libres sobre pocos datos habrían memorizado el corpus).

### Hiperparámetros (justificados)

La justificación de cada valor parte de una premisa: **tenemos un corpus chico (538 pares) y solo 0.27% de parámetros entrenables**, así que la configuración debe ser conservadora en capacidad, agresiva en regularización y barata en memoria.

- **`r = 8`.** Nuestro corpus es pequeño, así que el subespacio adaptativo no necesita ser grande: r=8 alcanza para que LoRA aprenda los patrones de relación consulta→norma sin darle a la rama la capacidad de memorizar los 538 pares. Un r mayor (32/64) solo habría añadido parámetros innecesarios sobre pocos datos, con más riesgo de overfitting y sin ganancia medible.
- **`lora_alpha = 16`.** Escala cuánto "pesa" la rama LoRA respecto al backbone congelado: con `α/r = 2` la actualización entra con fuerza suficiente para mover el comportamiento del modelo con solo 8 dimensiones, pero sin que un solo paso de descenso descarrile las representaciones pre-entrenadas. Si hubiéramos usado α=8 (factor 1), la adaptación habría sido demasiado tímida para el salto sobre el baseline.
- **`lora_dropout = 0.1`.** Con 538 pares el modelo tiende a sobreajustar; dropout en las ramas LoRA corta aleatoriamente conexiones de la adaptación en cada paso, forzando que la señal de relación se distribuya en varias dimensiones en lugar de depender de una. 0.1 es un valor estándar que regulariza sin llegar a dificultar el aprendizaje como lo haría 0.3-0.5.
- **`target_modules = ["query", "value"]`.** Las proyecciones query/value de la atención son las que deciden *qué mirar* de la consulta y de la norma; adaptar solo ahí es suficiente porque la tarea es entender qué partes de ambos textos se relacionan. Además es el patrón de menor riesgo de PEFT: tocar MLP o la cabeza de salida completa habría cuadruplicado los entrenables sin evidencia de beneficio en clasificación de pares.
- **`modules_to_save = ["classifier"]`.** La cabeza binaria se crea desde cero (no viene pre-entrenada): si no se incluyera en `modules_to_save`, sus pesos quedarían congelados en su inicialización aleatoria y el modelo no podría aprender nada. Se entrena aparte del backbone congelado.
- **`learning_rate = 1e-4`.** Como solo se actualizan 296K parámetros (las ramas A/B y la cabeza, todos "nuevos"), el modelo puede tolerar un paso más grande que el 2e-5 clásico de full fine-tuning, donde un paso grande corrompería los 110M de pesos pre-entrenados. Con lr menor el entrenamiento habría sido demasiado lento para nuestros ~192 pasos totales.
- **batch efectivo = 16** (8 × grad-accum 2). Con batch 8 el gradiente de cada paso sería muy ruidoso; con 16 se estabiliza la estimación del gradiente sin duplicar la memoria del T4. El grad-accum logra ese batch de 16 con el consumo de memoria de 8.
- **`warmup_steps = 19`.** ~10% de los ~192 pasos: al inicio las ramas A/B empiezan en valores arbitrarios (A aleatoria, B en ceros) y darles el lr completo desde el paso 1 produciría actualizaciones inestables. El warmup sube la tasa gradualmente mientras la rama se estabiliza.
- **`weight_decay = 0.01`.** Regulariza los pocos pesos entrenables (las ramas LoRA y la cabeza) para que no crezcan sin límite en un corpus chico; es el valor por defecto recomendado por AdamW y coherente con nuestra estrategia conservadora.
- **`num_train_epochs = 8`.** Tope superior generoso: dejamos que la combinación con early stopping decida cuándo parar (ver abajo). Con 8 épocas se garantiza que el F1 haya alcanzado su meseta antes de que el stop lo corte (terminó en 6).
- **`fp16` activo.** El T4 está optimizado para precisión mixta: ~2× más rápido de entrenamiento y la mitad de memoria de activaciones, sin pérdida de precisión observable en nuestras métricas. Sin él, el entrenamiento de 6 épocas habría tomado el doble en el límite del tiempo de Colab.
- **`max_length = 512` + `truncation = "only_second"`.** BETO tiene un tope duro de 512 tokens y nuestros artículos son largos (texto legal completo). Al truncar solo el segundo segmento, la consulta del abogado nunca se pierde y la norma se recorta al prefijo — aceptable porque los considerandos clave suelen estar al inicio del artículo.
- **`seed` / `data_seed = 42`.** Fija inicialización de pesos, orden del dataloader y particiones: otra persona al correr el notebook obtiene los mismos resultados. Sin esto, los números de la tabla serían irreproducibles.
- **`optim = adamw_torch`.** El weight decay desacoplado de AdamW es el estándar de Hugging Face: regulariza los pesos sin interferir con el momento adaptativo de cada parámetro, adecuado para nuestros pocos entrenables.

### Early stopping

La Trainer evalúa al final de cada época y guarda el mejor checkpoint por **F1 en validation** (`load_best_model_at_end`, `metric_for_best_model="f1"`). Con `EarlyStoppingCallback(patience=2)`, si F1 no mejora durante 2 épocas consecutivas se detiene el entrenamiento. De las 8 épocas configuradas se entrenaron **6**; el F1 alcanzó su máximo (0.763) en la **época 4** y se estabilizó:

| Época | Loss val | Acc | Prec | Recall | F1 |
|---|---|---|---|---|---|
| 1 | 0.701 | 0.494 | 0.538 | 0.156 | 0.241 |
| 2 | 0.648 | 0.621 | 0.875 | 0.311 | 0.459 |
| 3 | 0.546 | 0.782 | 0.933 | 0.622 | 0.747 |
| 4 | 0.496 | 0.793 | 0.935 | 0.644 | **0.763** |
| 5 | 0.447 | 0.793 | 0.935 | 0.644 | 0.763 |
| 6 | 0.468 | 0.793 | 0.935 | 0.644 | 0.763 |

### Resultados (validation)

| Modelo | Acc | Prec | Recall | F1 |
|---|---|---|---|---|
| Clase mayoritaria | 0.483 | 0.000 | 0.000 | 0.000 |
| BETO zero-shot | 0.483 | 0.000 | 0.000 | 0.000 |
| **BETO + LoRA** | **0.793** | **0.935** | **0.644** | **0.763** |

La tabla anterior es la **evaluación de clasificación**, fila por fila: por cada par (consulta, artículo) el modelo decide si es relevante o no — es la comparación oficial que pide la rúbrica (baseline vs. afinado sobre el mismo `validation`). Aparte, el **producto** funciona como recuperación: para cada consulta se ordenan los artículos candidatos por score y se devuelven los k mejores. Esa evaluación de **ranking** por sentencia (top-3) es la métrica principal del proyecto y también se calcula sobre `validation`:

Ránking de recuperación por sentencia (validation, k=3): **Recall@3 0.898** vs 0.679 de BETO sin afinar (zero-shot); **Precision@3 0.587** vs 0.381. (La clase mayoritaria figura como N/A en ranking: al dar el mismo score a todos los candidatos no define un orden.) En test, como información adicional del equipo: acc 0.895, prec 0.906, recall 0.853, F1 0.879.

**Lectura honesta.** F1 0.76 en validation refleja la dificultad real: recall 0.64 indica que aún se escapan positivos (normas que el modelo no relaciona). Precision 0.94 confirma que lo que marca como relacionado casi nunca es ruido — el asistente prefiere no afirmar antes que afirmar mal, aceptable para su rol de verificación. El delta frente al baseline (0 → 0.76 F1) es todo aporte del fine-tuning. Meta del proyecto (Recall@3 ≥ 0.85) **cumplida**.

### Ejemplos cualitativos

```
Consulta : "Trabajé para el ISS desde septiembre de 2000 hasta marzo de 2013 como contador
            público, pero me hicieron firmar contratos de prestación de servicios..."
Artículo : Ley 100 de 1993, Art. 275 (Instituto de Seguros Sociales)
Esperado : RELEVANTE   →   Predicho: RELEVANTE (score 0.781)  ✅

Consulta : "Me despidieron de Helicol después de 9 años trabajando, sin saber que estaba
            embarazada... la EPS me negó la licencia de maternidad..."
Artículo : CST Art. 239 (protección a la maternidad)
Esperado : NO RELEVANTE   →   Predicho: RELEVANTE (score 0.630)  ❌ falso positivo
```

Selección reproducible sobre validation (verdadero positivo representativo, falso negativo cercano al umbral, negativo difícil correctamente rechazado y falso positivo con mayor score) en la sección 8 del notebook. El score mostrado es la **score de relevancia**, no una confianza calibrada.

## Dataset

El corpus se construye en **dos fases** separadas: la **recolección** (de sentencias a pares pregunta→norma, en `notebooks/dataset/`) y la **adecuación para el modelo** (de pares a texto listo para tokenizar, al inicio del notebook de fine-tuning). Entre ambas hay un **join** con un diccionario de normas.

### Archivos CSV que componen el dataset

| Archivo | Rol | Contenido | Filas |
|---|---|---|---|
| `data/dataset_cross_encoder.csv` | **Dataset final de pares** (se consume en el entrenamiento) | `consulta`, `articulo` (cita, ej. `CST Art. 127`), `tipo`, `label` (1/0), `sentencia_origen` | 561 (138 sentencias) |
| `data/diccionario_articulos.csv` | **Texto normativo completo** (se une por llave) | `fuente`, `numero`, `texto_completo`, `n_citas_en_dataset`, `url_fuente` | 143 artículos / 41 normas |
| `data/fuentes_normas.csv` | **Insumo manual** del diccionario: URL oficial resuelta de cada norma | `fuente`, `url`, `dominio`, `notas` | 41 |
| `data/descartados.csv` | **Blacklist compartida** de sentencias descartadas | `sentencia`, `razon` | 1362 |
| `data/candidatas_research.csv` | **Insumo** de la vía de investigación externa | `sentencia`, `tema`, `url` | 130 |

### Pipeline de construcción (recolección → pares)

El pipeline completo se documenta paso a paso en esta sección y en los notebooks de `notebooks/dataset/`. En resumen:

1. **Recolección por 4 vías** (`notebooks/dataset/`): cada sentencia pasa un embudo de filtros antes de gastar llamadas LLM — prefiltro de keywords, **gate de pertinencia barato (Claude Haiku, solo el inicio del texto)** y **extracción completa (Claude Sonnet)** solo para las que pasan. Un LLM lee la sentencia y produce los hechos en lenguaje coloquial (la `consulta`) y los **artículos determinantes** (`positivo`). Cada positivo genera un **negativo fácil** (pool curado de tema distinto) y un **negativo difícil** (segunda llamada LLM: artículo real, temáticamente cercano, que no aplica). Las 4 vías comparten la blacklist de `descartados.csv` para no reprocesar sentencias ya vistas.
2. **Diccionario de artículos** (`ds_diccionario_articulos.ipynb`): extrae el texto íntegro de cada norma citada (CST, Constitución, ~40 leyes/decretos) desde los portales oficiales usando `fuentes_normas.csv` (URLs resueltas a mano).
3. **Join** (inicio del notebook de fine-tuning): la cita `CST Art. 127` se separa en la llave `(fuente, numero)` con normalización (NFKD: tildes, mayúsculas, espacios) y se une al diccionario para obtener `texto_input = "CST. " + texto_completo`. El texto del artículo NO vive embebido en cada fila: vive una sola vez en el diccionario (evita duplicar texto, sirve también en producción y centraliza correcciones).
4. **Limpieza**: se descartan 23 pares sin texto (21 citan una norma completa sin número de artículo, 2 sin fuente pública) → 538 utilizables.
5. **Split por sentencia** (`GroupShuffleSplit`, seed 42, dos etapas: 15% test, luego 15/85 del resto para validation): el positivo y sus negativos comparten la misma consulta, así que dividir por fila filtraría la consulta a varios splits (**fuga**). Split por sentencia lo impide.
6. **Tokenización**: pares a 512 tokens con `truncation="only_second"` (la consulta nunca se pierde; la norma se recorta al prefijo).

### Tamaño y split

561 pares pregunta→norma (285 positivos, 138 negativos fáciles, 138 negativos difíciles) → **538 utilizables** (labels 0:276 / 1:262):

| Partición | Pares | Sentencias | Positivos |
|---|---|---|---|
| Train | 375 | 96 | 183 |
| Validation | 87 | 21 | 45 |
| Test | 76 | 21 | 34 |

**Idioma:** español. **Licencia:** textos normativos y sentencias de dominio público (fuentes oficiales del Estado colombiano); los pares de entrenamiento (extracción y negativos generados por LLM) son material propio del equipo. Código: MIT.

## Reproducibilidad y equipo

Reproducible de punta a punta en `notebooks/finetuning/Fine_tuning_Asistente_legal.ipynb` (salidas preservadas): seed 42, REF de datos = `main` (commits fijados en la entrega), adaptador LoRA `s04_lora_adapter` y gráficas en la carpeta de salida. Requiere `pip install -r requirements.txt`; las claves API solo se usan en la recolección del dataset (`notebooks/dataset/`).

**Equipo — "Lawten"**: Pablo Cabrejos, Miguel Ángel Ortiz, Martín Valencia, Samuel López.
