# M3 — Checklist de entrega

Responsable de consolidación: Samuel López. Estado inicial: 29 de octubre de 2026.
Fuente: lista «Antes de entregar» de `STAI_M3_Asignacion_RAG_agentico.pdf`, páginas 5–6,
y reparto de `M3_PLAN_EQUIPO.pdf`, versión 2, secciones 7–9.
Los PDF fueron proporcionados al equipo y no se agregan al repositorio.

Una casilla marcada acredita la evidencia indicada; no significa que se haya repetido una corrida en Colab.
La columna de ejecución real y sus pendientes están en el [informe de reproducibilidad](docs/m3/REPRODUCIBILIDAD.md).

- [ ] El pipeline de indexación corre de principio a fin en Colab gratuito, sin errores. **Verifica:** Samuel; **responsable técnico:** Pablo. **Falta:** ejecutar en un runtime limpio y guardar evidencia de reconstrucción de Chroma. Las salidas anteriores no cierran esta verificación.
- [ ] Justificamos el tamaño y el solapamiento de los chunks, el modelo de embeddings y el top-k con argumentos de nuestro corpus. **Escribe:** Pablo; **verifica:** Samuel. **Falta:** completar «Tres decisiones justificadas» del README; la configuración ya está inventariada.
- [ ] El RAG ingenuo sigue en el repositorio, con válvula de escape, como línea base. **Responsable técnico:** Pablo; **verifica:** Samuel. **Evidencia parcial:** `notebooks/rag/rag_ingenuo.ipynb` y su scorecard. **Falta:** cerrar la revisión de la válvula: el sistema siempre muestra tres referencias y el reconocimiento de límites depende del decoder, como declara el README.
- [x] Integramos al menos dos técnicas avanzadas y dijimos qué falla ataca cada una. **Responsable:** Pablo; **verifica:** Samuel. **Evidencia versionada:** híbrida y reranking en `notebooks/rag/rag_avanzado.ipynb`, sus CSV de ablación y el análisis en «Ablación del RAG avanzado» del README. La nueva corrida limpia sigue pendiente en el primer ítem.
- [ ] Integramos al menos una herramienta, con esquema, parseo tolerante a errores y tope de pasos. **Entrega:** Miguel; **verifica:** Samuel. **Falta:** `notebooks/rag/agente_herramientas.ipynb` y evidencia de esos controles.
- [ ] Mostramos consultas en las que la herramienta se usa y al menos una en la que no hace falta. **Entrega:** Miguel; **verifica:** Samuel. **Falta:** al menos dos ejemplos con llamada y uno sin ella, con salidas del notebook.
- [ ] La tabla evolutiva está completa, sobre el mismo eval set, con la latencia mediana. **Consolida:** Pablo; **entrega resultados finales:** Miguel; **verifica:** Samuel. **Evidencia parcial:** tres filas actuales del README respaldadas por los scorecards. **Falta:** fila del sistema final y sus dos CSV bajo `notebooks/rag/resultados/final/`.
- [x] El eval set tiene al menos 10 casos, y al menos 2 tienen la respuesta fuera del corpus. **Responsable:** Pablo; **verifica:** Samuel. **Evidencia:** `notebooks/evaluation/eval_set.json` tiene 13 casos; M2-11 es premisa falsa y M2-12 es fuera de dominio. Ambos esperan reconocimiento de límites. No se modificó el eval set.
- [ ] Calculamos las cuatro métricas de RAGAS, con contexto y referencia por caso, y dijimos qué juez usamos. **Entrega:** Martín; **verifica:** Samuel. **Evidencia parcial:** PR #1 con ingenuo y avanzado. **Falta:** integrar el entregable, añadir evaluación del sistema final, comprobar el notebook ejecutado y completar la tabla desde su CSV.
- [ ] La lectura responde qué movió qué, cuánto costó, qué falla queda (recuperación o generación) y qué pasó con la válvula. **Escribe:** Pablo, con insumos de Martín y Miguel; **verifica:** Samuel. **Falta:** completar «Lectura honesta final» con los resultados pendientes. Se conserva la lectura existente de las tres versiones medidas.
- [x] El README abre con una frase entendible por alguien sin formación en IA. **Responsable y verificador:** Samuel. **Evidencia:** primer párrafo del README: usuario, tarea y criterio de buena respuesta.
- [ ] El README tiene la sección «Uso de IA» (o dice en una línea que no usamos ninguna). **Declaran:** los cuatro integrantes; **consolida:** Samuel. **Evidencia parcial:** sección creada con el uso observado en la preparación de Samuel. **Falta:** declaraciones del resto y confirmación de revisión humana de Samuel; no se inventan aportes personales.
- [ ] Fijamos semillas y versiones, y el repositorio queda ordenado y reproducible. **Responsables técnicos:** dueños de cada notebook; **verifica:** Samuel. **Evidencia parcial:** semillas y archivos requeridos revisados estáticamente; un solo `requirements.txt` en esta rama. **Falta:** revisar el agente, resolver diferencias de dependencias del PR de Martín, registrar el entorno real de Colab y confirmar el alcance de la autorización del juez externo. Ver informe.

## Entregables de Samuel adelantados

| Entregable | Evidencia | Estado |
|---|---|---|
| Esqueleto M3, frase inicial y diagrama | [README.md](README.md) | Preparado; integración definitiva del agente pendiente |
| Script y tabla del corpus | `scripts/resumen_corpus.py`, tabla del README y [salida JSON](docs/m3/corpus_resumen.json) | Ejecutado localmente; no modifica datos |
| Checklist con responsables | Este archivo | Preparado; pendientes explícitos |
| Informe de notebooks y controles | [REPRODUCIBILIDAD.md](docs/m3/REPRODUCIBILIDAD.md) y sus evidencias JSON | Revisión estática hecha; Colab limpio pendiente |
| Declaración de IA | Sección «Uso de IA» del README | Parcial; requiere aportes personales |

## Pendientes para que Samuel pueda cerrar

| Quién | Qué debe quedar disponible | Qué verifica o integra Samuel |
|---|---|---|
| Miguel | Agente, herramienta definida, ejemplos, notebook ejecutado y scorecards del sistema final | Completar el diagrama e índice y probar el notebook en Colab limpio |
| Martín | RAGAS integrado y ejecutado, CSV y diagnóstico incluyendo el sistema final | Completar tabla desde CSV, revisar juez, semillas y ejecución |
| Pablo | Tres decisiones justificadas, tabla evolutiva final y lectura honesta final | Revisar presencia, trazabilidad de cifras y coherencia con los CSV |
| Pablo y Martín | Acuerdo sobre dependencias y alcance de la excepción del juez por API | Registrar la decisión y probar las instrucciones resultantes |
| Todos | Declaración personal de uso de IA y revisión humana | Consolidar la sección sin atribuir revisiones no confirmadas |
| Samuel | Runtimes limpios de Colab, acceso a Secrets del juez y salidas completas de los notebooks integrados | Guardar evidencia de cada corrida y actualizar este checklist y el informe |

Los cambios se proponen desde `m3/samuel-documentacion`; su integración a `main` queda para un PR revisado por Pablo.
