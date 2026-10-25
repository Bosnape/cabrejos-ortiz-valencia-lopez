# Reporte Técnico de Evaluación RAGAS — Entrega M3

---

## 1. Tabla Consolidada de Resultados

Métricas calculadas sobre los 13 casos del eval set congelado (`notebooks/evaluation/eval_set.json`), utilizando `claude-haiku-4-5` como LLM-as-a-judge vía la API de Anthropic y evaluando los textos completos del diccionario (`data/diccionario_articulos.csv`):

| Sistema | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
| :--- | :---: | :---: | :---: | :---: |
| **RAG Ingenuo** (denso MiniLM, top-3) | 0.3197 | 0.1862 | 0.6431 | 0.3400 |
| **RAG Avanzado** (C2: híbrida + rerank) | 0.2302 | **0.2438** | **0.7608** | **0.4038** |
| **Sistema Final** (capa agéntica) | *Pendiente* | *Pendiente* | *Pendiente* | *Pendiente* |

*Nota: El detalle caso a caso con cada una de las 13 consultas se encuentra en `notebooks/rag/resultados/ragas/ragas_m3.csv`.*

---

## 2. Decisiones Metodológicas y Criterios de Evaluación

1. **Aislamiento de la Generación (`respuesta`):**  
   Se evaluó estrictamente el texto del `Resumen:` redactado por `Qwen2.5-1.5B-Instruct`. Se excluyó la lista de referencias numeradas y el aviso legal final, dado que ambos bloques son concatenados por código determinista en Python y no constituyen generación del modelo de lenguaje.

2. **Contextos Completos (`contextos`):**  
   Se alimentó al evaluador con el campo `texto_completo` de `data/diccionario_articulos.csv` correspondiente a cada una de las citas de `articulos_recuperados`. Se descartaron las descripciones truncadas a 400 caracteres que ve el decoder, con el fin de diagnosticar si el artículo legal en su totalidad contenía la solución al problema.

3. **Verdad Terreno (`referencia`):**  
   Se tomó el campo `expected` de cada caso del eval set como referencia fáctica y sustantiva.

4. **Juez Evaluador:**  
   Se utilizó `claude-haiku-4-5` de forma directa y determinista con temperatura 0 y validación estricta de esquemas JSON, garantizando paridad con el juez del harness de M2.

---

## 3. Lectura Diagnóstica y Diagnóstico Desacoplado

### 3.1. Separación de Fallas: Recuperación vs. Generación

* **Diagnóstico de Recuperación (Retriever):**  
  El `context_recall` promedio se ubica en 0.1862 para el RAG ingenuo y en 0.2438 para el RAG avanzado. En la gran mayoría de casos estándar (M2-02, M2-03, M2-04, M2-07, M2-08), los artículos que contienen la norma esperada (`expected`) nunca llegaron al top-3 de candidatos entregados al decoder. Esto demuestra empíricamente que la principal limitación del asistente jurídico reside en la **etapa de recuperación** (el retriever no logra asociar el relato fáctico del usuario con la norma sustantiva aplicable), y **no** en el generador.

* **Diagnóstico de Generación (Decoder):**  
  La métrica de `faithfulness` aumenta significativamente en el RAG avanzado, alcanzando **0.7608** frente a 0.6431 del ingenuo. Qwen2.5-1.5B demuestra alta disciplina para resumir exclusivamente lo que dicen los fragmentos entregados, evitando inventar normas ajenas. Su bajo acierto en la evaluación de negocio de M2 se debe a que opera sobre contextos que carecen de la norma pertinente, y no a alucinación activa.

* **Relevancia Temática (`answer_relevancy`):**  
  La relevancia de respuesta sube de 0.3400 a 0.4038. Al recibir artículos con mayor coherencia temática general gracias a la fusión híbrida y el cross-encoder, los resúmenes abordan con mayor propiedad los términos laborales de la consulta.

### 3.2. Contraste entre `recall_3` del Harness y `context_recall` de RAGAS

* El `recall_3` del harness M2 es **léxico y binario**: exige la coincidencia exacta de la cita (por ejemplo, `CST Art. 239`). En el caso M2-01 (despido durante embarazo), el sistema recuperó la `Ley 1822 de 2017 Art. 2`, la cual modifica y transcribe el contenido del Art. 239 del CST. El harness penalizó esto con un recall de **0.0**.
* En contraste, el `context_recall` de RAGAS es **semántico y sustancial**: Claude Haiku verificó que la protección reforzada a la maternidad y la prohibición de despido sin autorización del Ministerio de Trabajo sí estaban consagradas en el artículo recuperado de la Ley 1822, otorgando en M2-01 un `context_recall` de **1.00**.

### 3.3. Análisis de Casos Especiales y de Abstención

* **M2-12 (Fuera de dominio — accidente de tránsito):**  
  En el RAG avanzado, el decoder reconoce explícitamente que las normas recuperadas no tratan accidentes de tránsito, logrando un puntaje de relevancia de **0.92** en su función de abstención.
* **M2-11 (Premisa falsa — auxilio ficticio por CST 850):**  
  El sistema trae artículos generales y el resumen no logra desmentir la premisa falsa del usuario, arrojando baja relevancia (0.25). Esto ratifica el hallazgo documentado en el proyecto: la válvula de escape no es una decisión gobernada por el sistema, sino una conducta dependiente de las capacidades del decoder.

---

## 4. Estado de los Entregables

1. Cuaderno ejecutable: `notebooks/rag/ragas_m3.ipynb` (probado end-to-end, reproducible en Google Colab con GPU T4).
2. Tabla de métricas exportada: `notebooks/rag/resultados/ragas/ragas_m3.csv`.
3. Integración futura: El cuaderno contiene la función modular `evaluar_sistema_ragas` lista para incorporar el archivo `scorecard_sistema_final_detalle.csv` de Miguel una vez sea generado.
