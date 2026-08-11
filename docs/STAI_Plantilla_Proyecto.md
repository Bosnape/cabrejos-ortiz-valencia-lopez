# Proyecto integrador STAI — Plantilla de definición

---

## 0 · Equipo

- **Nombre del equipo:** Lawten
- **Integrantes** (nombre · correo):
  - Pablo Cabrejos · pcabrejosm@eafit.edu.co
  - Miguel Angel Ortiz · maortizp@eafit.edu.co
  - Martin Valencia · mvalenciv1@eafit.edu.co
  - Samuel Lopez · slopezg16@eafit.edu.co
- **Integrante de contacto con el profesor:** Pablo Cabrejos
- **Cómo van a coordinar el código** (GitHub / GitLab · enlace al repo): GitHub · https://github.com/Bosnape/cabrejos-ortiz-valencia-lopez
- **Día y hora semanal de reunión fuera de clase:** Viernes a las 3:00 pm

---

## 1 · Tema y proyecto

**En una frase, ¿qué es su proyecto?**

Un asistente de consulta normativa en derecho laboral individual colombiano, dirigido a abogados junior: dado lo que el abogado pregunta o escribe, el sistema identifica qué norma aplica y le entrega la cita exacta.

**¿Por qué este tema?**

Porque el error que resuelve es real y costoso: un abogado junior que cita la norma equivocada perjudica un caso de verdad, y hoy esa verificación se hace de memoria o con búsquedas manuales lentas. El sistema no reemplaza el juicio jurídico del abogado — solo hace que encontrar la norma correcta deje de tomar minutos.

---

## 2 · Usuario y decisión

- **¿Quién usaría este sistema?** Un abogado junior (entre cero y tres años de experiencia) mientras redacta un documento: una demanda, una contestación, un concepto o una cláusula contractual.
- **¿Qué decisión o tarea concreta resuelve con la ayuda del sistema?** Qué norma citar o aplicar a la situación descrita. El sistema se limita a informar y verificar normas — no recomienda acciones sobre el caso ni sobre el cliente; el juicio jurídico lo conserva el abogado.
- **¿Qué hace hoy esa persona sin el sistema?** Busca de memoria o hace búsquedas manuales lentas en la norma, o usa herramientas conversacionales genéricas que pueden entregar información inexacta.

**Dominio y alcance:** normativa laboral **individual** colombiana — la vida jurídica de un contrato de trabajo, desde su formación hasta su terminación, incluyendo las protecciones especiales que la ley y la jurisprudencia reconocen a trabajadores en situación de vulnerabilidad. Seis áreas: **contratación** (tipos de contrato, presunción de la relación laboral, contrato realidad); **jornada y recargos** (horas extra, recargo nocturno, dominicales y festivos); **terminación y liquidación** (despido con y sin justa causa, prestaciones al finalizar el contrato); **estabilidad laboral reforzada** (embarazo, discapacidad o enfermedad, prepensionados, madres y padres cabeza de familia — casos en que la terminación del contrato exige autorización especial); **discriminación laboral** en el empleo; y **prestaciones sociales** (cesantías, vacaciones, licencias de maternidad/paternidad/calamidad doméstica). Se cubren también, de forma más acotada, el *ius variandi* (modificación unilateral de las condiciones de trabajo por el empleador) y el acoso laboral (Ley 1010 de 2006), cuando el punto de derecho central es la relación laboral individual.

Quedan fuera el derecho laboral colectivo (sindicatos, fuero, huelga) y la seguridad social (pensiones, afiliación y prestaciones de salud) — con una excepción puntual: cuando una EPS, AFP u otra entidad del sistema de seguridad social aparece como **empleadora demandada** en un litigio de salario, despido o prestaciones de uno de sus propios trabajadores, ese caso sí es laboral individual y entra en el alcance; deja de aplicar la exclusión solo cuando el litigio versa sobre la prestación de salud o pensión en sí misma.

---

## 3 · Tarea central del modelo (M1)

- **¿Qué tipo de tarea es?** Un cross-encoder: recibe un par de textos — la consulta del abogado y un artículo normativo candidato — y produce un score de relevancia formal entre 0 y 1, indicando qué tan bien ese artículo responde esa consulta específica. No es clasificación de un solo texto ni generación: procesa los dos textos juntos y produce un solo número.

- **¿Cuál es el input? ¿Cuál es el output esperado?**
  ```
  Input:  "me despidieron sin avisar, ¿me deben algo?" [SEP] "CST Art. 64 - indemnización por terminación sin justa causa"
  Output: 0.93 (muy relevante)

  Input:  "me despidieron sin avisar, ¿me deben algo?" [SEP] "CST Art. 161 - jornada máxima legal"
  Output: 0.04 (no relevante)
  ```
  Este score se usa para reordenar los candidatos que trae un retrieval por similitud semántica (embeddings, sin fine-tuning) sobre todo el corpus normativo, y quedarse con el artículo (o los 2-3 artículos) que realmente aplican.

  La dificultad que justifica el ajuste fino: las consultas llegan en lenguaje coloquial ("me echaron", "¿me pueden bajar el sueldo?"), sin terminología jurídica, y distinguir entre artículos temáticamente parecidos (dos artículos sobre terminación, pero con causales distintas) requiere entender la consulta a un nivel más fino que la similitud semántica genérica de un modelo sin entrenar en el dominio.

- **¿Qué modelo base?** encoder `dccuchile/bert-base-spanish-wwm-cased` (BETO) — BERT monolingüe en español (Whole Word Masking). Elegido tras comparar tokenización de dominio contra alternativas multilingües: BETO fragmenta menos el vocabulario legal/laboral en español (fertilidad de 1,35 tokens por palabra frente a 1,56 en mBERT/DistilBERT), lo que se traduce en ~14% menos tokens sobre el mismo texto. Corre sin problema en el T4 gratis de Colab, con margen para LoRA o incluso fine-tuning completo dado el tamaño (ver `notebooks/decisiones_base_encoder.ipynb`).

---

## 4 · Dataset

- **¿De dónde sale el texto?**
  - Pares de entrenamiento: **no son sintéticos** — cada consulta sale de una sentencia real (tutelas T/SU de la Corte Constitucional y providencias SL de la Sala de Casación Laboral de la Corte Suprema, más una base curada de redal.org). Un LLM lee el texto completo de la sentencia y extrae los hechos en lenguaje coloquial, tal como los contaría el trabajador afectado, junto con los artículos que fueron determinantes para la decisión (`positivo`). Por cada positivo se agrega un negativo fácil (artículo de un tema claramente distinto, de un pool fijo) y un negativo difícil (un artículo real, generado por el mismo LLM, temáticamente cercano al correcto pero que no aplica al caso). Cada sentencia pasa antes por un filtro de pertinencia en dos etapas (un gate barato con Claude Haiku, extracción completa con Claude Sonnet solo para las que pasan) y por auditoría manual del equipo.
  - Textos normativos (diccionario de artículos, para dar contexto completo a cada candidato en el ranking): texto íntegro de cada norma efectivamente citada por las sentencias del corpus — CST, Constitución Política, y ~40 leyes/decretos/resoluciones específicos según el caso, no un conjunto fijo elegido de antemano. Obtenido de fuentes oficiales (`funcionpublica.gov.co`, `alcaldiabogota.gov.co` — Régimen Legal, repositorio nacional pese al nombre). Ver `data/README.md` para el detalle completo.
  - Golden set de evaluación: aún por construir, separado del conjunto de entrenamiento.

- **¿Cuántos ejemplos, aproximadamente?** 138 sentencias únicas → 561 pares (285 positivos, 138 negativos fáciles, 138 negativos difíciles). Corpus cerrado deliberadamente en este punto — por debajo del rango inicialmente estimado, con rendimientos decrecientes en las últimas rondas de ampliación.

- **¿En qué idioma(s) está?** Español.

- **¿Hay alguna licencia que restrinja el uso?** Los textos normativos y las sentencias son de dominio público (fuentes oficiales del Estado colombiano). Los pares de entrenamiento (extracción y negativos generados por LLM) son material propio del equipo.

- **¿Algún sesgo o limitación conocida del dataset?** El corpus depende de qué llega a instancias judiciales superiores, lo que sesga estructuralmente su composición: sobrerrepresenta estabilidad laboral reforzada (embarazo, salud, discapacidad — los casos que más llegan a tutela) y subrepresenta jornada/recargos y factores salariales (arts. 127-132 y 158-179 CST), cuyas disputas rara vez escalan más allá de la vía ordinaria. Vacaciones (arts. 186-192 CST) está prácticamente en cero pese a haberse probado varios canales de búsqueda — puede ser un hueco estructural real, no solo de cobertura de búsqueda.

---

## 5 · Métrica de éxito

- **¿Qué métrica principal medirá si el sistema es útil?** Recall@k y Precision@k del cross-encoder sobre el golden set (k=3 o k=5, por definir junto con Miguel/Samuel) — de los k artículos con mayor score después del reordenamiento, ¿qué fracción de los realmente aplicables se recuperó (recall) y qué fracción de lo devuelto es correcto (precision)? Meta ≥ 0,85 en Recall@k.

- **¿Por qué esa métrica y no otra?** El dataset es multi-etiqueta — varias consultas tienen más de un artículo correcto a la vez (285 positivos sobre 138 sentencias), y el sistema está pensado para devolver varios candidatos, no uno solo. Accuracy@1/MRR asumen una única respuesta correcta y solo evalúan si quedó primera; penalizan mal un caso donde el modelo recupera varios artículos válidos entre los primeros k pero no el que se marcó como "el" target. Recall@k/Precision@k mide directamente si el conjunto de artículos que se le muestra al abogado es el correcto.

- **¿Cuál sería un baseline razonable con el que compararse?** El mismo modelo sin fine-tuning (zero-shot) haciendo el mismo reordenamiento, o el retrieval por similitud semántica solo, sin el paso de reordenamiento. El fine-tuning solo se justifica si supera claramente ese baseline.

**Señal de valor real:** en una prueba con usuarios, el abogado confirma la cita en segundos en lugar de minutos, y en al menos el 70% de los casos la respuesta le basta sin una búsqueda manual adicional.

---

## 6 · Componente visual (M4) — plan inicial

- **¿Cómo podría integrarse un componente visual a este sistema?** El abogado aporta la imagen de un documento — un contrato, una demanda o un concepto, escaneado o fotografiado — y un modelo de visión-lenguaje interpreta la imagen directamente, sin reducirla a texto plano primero. La tarea concreta: detectar qué normas cita el documento y verificar, contra el corpus normativo, que esas citas son correctas (el artículo existe y dice lo que el documento afirma que dice). Como línea base de comparación, el mismo resultado puede intentarse con OCR más el pipeline de texto existente; el componente visual se justifica si interpretar la imagen directamente mejora la detección frente a esa línea base.

- **¿Tienen ejemplos de imágenes representativas?** Por definir — fotos o escaneos de contratos laborales y cartas de despido reales o generados como ejemplo.

---

## 7 · Riesgos éticos y de uso

- **¿Quién podría salir perjudicado si el sistema falla?** El cliente del abogado, si el sistema indica una norma que no aplica realmente al caso y el abogado la cita en un documento real.
- **¿Hay grupos de personas en los que probablemente falle más?** Consultas sobre temas subrepresentados en el corpus (jornada/recargos, factores salariales, vacaciones — ver sesgo en sección 4), o redactadas de forma muy distinta a como se relatan los hechos en una sentencia.
- **¿Cómo se mitigarían esos riesgos?** El sistema se limita a verificar y citar normas, nunca a aconsejar acciones sobre el caso. Cada respuesta muestra la fuente y el artículo exacto. Ante la duda, el sistema declara explícitamente que no encontró la información o que requiere verificación humana, en lugar de completar con datos inciertos. El dataset se audita manualmente por rondas (estructura, límites de alcance, formato de citas) antes de darlo por cerrado. Se advierte que el sistema no sustituye el juicio profesional del abogado.

---

### Checklist rápido antes de cerrar

- [x] El tema tiene texto real de un dominio con el que podemos trabajar.
- [x] Sabemos quién es el usuario y qué decisión concreta toma.
- [x] Tenemos al menos una fuente de datos identificada.
- [x] Tenemos una métrica principal en mente.
- [x] Todos en el equipo pueden ejecutar un Colab con GPU.
