# M3 — Informe de verificación de Samuel

Fecha de revisión: 29 de octubre de 2026 (America/Bogota).
Base examinada: `main`, commit `426a70aceb85655f1b5e962116e6cd0ec608c107`.
Trabajo de documentación: rama `m3/samuel-documentacion`.
RAGAS examinado por separado: `origin/m3/martin-ragas`, commit `c8b8f1d5003fead3077acb09688e616a242798e9`.

## Alcance y resultado

Se ejecutaron controles locales de inventario y revisión estática con Python, sin instalar el stack de modelos,
reentrenar BETO, generar respuestas ni consumir la API del juez. No se modificaron `data/`,
`notebooks/evaluation/`, los notebooks RAG base, el adaptador LoRA ni los CSV de resultados.

**No se ha realizado una nueva ejecución de punta a punta en Colab limpio.** La sesión local de revisión
no tiene configurada `ANTHROPIC_API_KEY` ni un `.env`; las corridas completas del juez quedan pendientes.
Esto no se clasifica como un fallo de los notebooks. Una revisión de código o de salidas antiguas
no permite afirmar que funcionen hoy en un runtime nuevo.

## Evidencia y comandos reproducibles

Desde la raíz del checkout, con Python 3 y Git:

```bash
python scripts/resumen_corpus.py
python scripts/resumen_corpus.py --json
python scripts/verificar_m3.py
python scripts/verificar_m3.py --ref origin/m3/martin-ragas
```

Los scripts usan la biblioteca estándar. El primero valida columnas, campos requeridos y unicidad
de `(fuente, numero)`, cuenta los CSV sin modificarlos y emite las huellas de los insumos.
El segundo lee los notebooks como JSON y analiza la sintaxis de sus celdas Python; no las ejecuta.
`--ref` lee una revisión de Git sin cambiar de rama ni incorporar archivos de otro integrante.
La referencia remota debe haberse descargado previamente con `git fetch origin`.

| Evidencia guardada | Qué permite comprobar |
|---|---|
| [corpus_resumen.json](corpus_resumen.json) | 41 normas, 143 artículos/chunks previstos, 138 sentencias y 561 pares; fuentes y huellas de los CSV |
| [verificacion_estatica.json](verificacion_estatica.json) | Estado del núcleo actual, archivos requeridos, semillas, contadores de ejecución y errores guardados |
| [verificacion_martin.json](verificacion_martin.json) | Revisión separada de la propuesta de RAGAS en el commit indicado arriba |

Las huellas SHA-256 normalizan el texto a UTF-8 sin BOM y saltos LF, para no confundir conversiones
de saltos de línea de Windows con cambios de contenido. Los conteos de chunks se derivan de
la indexación actual de un artículo por unidad; no prueban el tamaño de una colección Chroma reconstruida.
Las evidencias JSON son una fotografía de esta revisión; se deben regenerar al integrar nuevas versiones.

## Estado por notebook

La numeración de celdas en los JSON empieza en 1 e incluye las celdas Markdown.

| Notebook y origen | Comprobación estática | Salidas versionadas | Nueva corrida en Colab limpio |
|---|---|---|---|
| Harness de M2, base de comparación, `main` | 18 celdas de código con sintaxis válida; archivos requeridos presentes; semilla 42 | 18 celdas con contador, sin salidas de tipo `error` guardadas | Pendiente |
| RAG ingenuo, `main` | 17 celdas de código con sintaxis válida; archivos requeridos presentes; semilla 42 | 17 celdas con contador, sin salidas de tipo `error` guardadas | Pendiente |
| RAG avanzado, `main` | 23 celdas de código con sintaxis válida; archivos requeridos presentes; semilla 42 | 23 celdas con contador, sin salidas de tipo `error` guardadas | Pendiente |
| Agente de Miguel | El archivo esperado `notebooks/rag/agente_herramientas.ipynb` está ausente en las revisiones examinadas | Pendiente de entrega | Pendiente de Miguel y de prueba posterior |
| RAGAS de Martín, rama `m3/martin-ragas` | 8 celdas de código con sintaxis válida; archivos requeridos presentes; `SEED = 42`, semillas de Python y NumPy | Sin contadores ni outputs guardados en el notebook de esa revisión; sí existe el CSV de métricas en su rama | Pendiente de notebook ejecutado e integración; después falta el sistema final |

Las llamadas a semillas se comprueban como declaraciones en el código. No garantizan determinismo del proveedor
remoto. Los contadores y la ausencia de errores guardados tampoco prueban que una corrida nueva esté completa.
En RAGAS se evalúan resultados guardados con un juez remoto; su script no carga Torch, por lo que no se exige
una llamada a `torch.manual_seed` para esa parte.

## Hallazgos que requieren cierre

| Hallazgo | Evidencia | Responsable y siguiente acción |
|---|---|---|
| Falta el sistema final y su latencia | No están el notebook del agente ni los dos CSV de `resultados/final/` en las ramas examinadas | Miguel entrega; Pablo consolida la tabla; Samuel verifica e integra enlaces y diagrama |
| RAGAS aún no está integrado y falta evaluar el sistema final | PR #1 de Martín; su CSV actual cubre ingenuo y avanzado | Martín entrega la versión final ejecutada y su diagnóstico; Samuel copia la tabla desde el CSV integrado |
| Dependencias diferentes entre la propuesta de Martín y sus instrucciones de Colab | `requirements.txt` de su rama fija `anthropic==1.4.0`, `pandas==3.0.5`, `python-dotenv==1.2.3`; su bootstrap instala `2.3.0`, `3.0.3`, `1.2.2`, respectivamente | Martín y Pablo acuerdan las instrucciones antes de integrar; Samuel verifica la ejecución. No se alteró ese trabajo desde esta rama |
| El plan pide un `requirements.txt` raíz sin versiones | Se conserva así en esta rama; los bootstraps tienen versiones propias | Pablo confirma el acuerdo de instalación; Samuel registra las versiones realmente usadas en Colab |
| API de pago frente al requisito de alternativa sin ella | La asignación pide alternativa sin API obligatoria; el plan y el README declaran Haiku autorizado por el profesor | Pablo confirma y documenta si la autorización cubre esa excepción; si no, el equipo resuelve la alternativa. Samuel no cambia el juez ni la evaluación |
| La válvula depende del resumen del decoder | Limitaciones del README y casos M2-11/M2-12; el sistema siempre presenta tres referencias | Pablo cierra la lectura; Miguel y Martín aportan resultados finales. Samuel mantiene el ítem pendiente hasta contar con evidencia y decisión |
| Las revisiones exactas de los modelos y el entorno limpio no están certificadas por esta revisión | Los notebooks declaran identificadores de modelos; aquí no se descargaron ni se ejecutaron | Registrar revisiones descargadas y versiones del entorno en las corridas reales, sin cambiar los modelos por cuenta propia |

## Controles locales adicionales

El informe estático registra un solo `requirements.txt` versionado en la raíz de esta rama,
`.env` excluido por `.gitignore` y no versionado, y `.env.example` presente.
Se revisaron los archivos de texto versionados de las revisiones indicadas con patrones de claves
Anthropic, tokens de GitHub y claves privadas, sin coincidencias. Es un control de patrones conocidos,
no una auditoría de todo el historial ni una garantía sobre cualquier formato de credencial.

El eval set contiene 13 casos y los dos casos de abstención M2-11 y M2-12.
Los modelos, el top-k y las semillas declaradas del núcleo constan en los JSON.
No se corrigió la lógica de otro integrante ni se cambiaron las reglas de acierto.

También se comprobó que los conteos coinciden con la tabla del README, que el script del corpus funciona
desde otro directorio, que sus salidas repetidas y las del verificador son idénticas para los mismos insumos,
y que los enlaces locales y bloques Markdown de la documentación están completos.
Los tres scorecards existentes conservan 13 filas y 2, 1 y 2 aciertos respectivamente.
`git diff --check` no reportó errores de espacios y la comparación con `origin/main` no mostró cambios
en datos, notebooks, adaptador, resultados, dependencias o configuración de secretos.

## Cómo completar las pruebas de Colab

1. Identificar el commit a probar y abrir o subir el notebook correspondiente en una sesión nueva de Colab.
   El bootstrap de los notebooks puede clonar `main` para leer insumos; registrar ese commit también.
   Para notebooks aún fuera de `main`, subir su `.ipynb`, como indica el plan del equipo.
2. Seleccionar GPU T4, verificar la salida `device: cuda` donde corresponda y habilitar el secreto
   `ANTHROPIC_API_KEY` sin incluir su valor en capturas, celdas ni archivos versionados.
3. Ejecutar todas las celdas en orden. Verificar la reconstrucción de Chroma en los notebooks de recuperación,
   la carga de los mismos 13 casos y la exportación de los scorecards o métricas esperados.
4. Guardar el notebook ejecutado y la evidencia del entorno: fecha, commit del notebook e insumos,
   GPU, versión de Python, paquetes instalados, revisiones de modelos y salida final.
   Conservar aparte las nuevas evidencias; no sobrescribir los CSV congelados del núcleo.
5. Si falla, registrar celda, mensaje y entorno, y trasladar el hallazgo al dueño del notebook.
   Samuel no corrige su lógica ni cambia el corpus, el eval set, el decoder, el prompt o el juez para pasar la prueba.
6. Repetir para ingenuo, avanzado, agente y RAGAS, y revisar la base de comparación si cambia su entorno.
   RAGAS del sistema final solo se cierra después de disponer del detalle generado por Miguel.
7. Actualizar la tabla de este informe y [M3_CHECKLIST.md](../../M3_CHECKLIST.md) con enlaces a la evidencia.
   Una corrida nueva puede variar en latencia y puntajes del juez; documentar la variación, sin inventar igualdad exacta.

## Registro pendiente de ejecución real

| Notebook | Commit probado | Fecha / GPU | Evidencia de Run all | Resultado o error | Responsable |
|---|---|---|---|---|---|
| RAG ingenuo | Pendiente | Pendiente | Pendiente | No probado en runtime nuevo | Samuel |
| RAG avanzado | Pendiente | Pendiente | Pendiente | No probado en runtime nuevo | Samuel |
| Agente | Pendiente de Miguel | Pendiente | Pendiente | No disponible en las revisiones examinadas | Miguel entrega; Samuel verifica |
| RAGAS | Pendiente de integración | Pendiente | Pendiente | CSV existente no sustituye el notebook ejecutado | Martín entrega; Samuel verifica |

Además de estas pruebas, faltan las declaraciones personales de IA, las tres justificaciones y la lectura final de Pablo.
El [checklist](../../M3_CHECKLIST.md) concentra esas dependencias para cerrar la parte de Samuel.
