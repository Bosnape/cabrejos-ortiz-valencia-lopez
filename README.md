# Asistente de Consulta de Derecho Laboral Individual (SI4006)

Asistente de apoyo para abogados junior colombianos: verifica qué norma/artículo aplica a
una consulta de derecho laboral individual. Informa y verifica — nunca recomienda acciones
sobre un caso concreto.

## Estructura

- `notebooks/decisiones_base_encoder.ipynb` — elección de familia/modelo base (BETO),
  tokenización de dominio, justificación técnica
- `notebooks/dataset/` — recolección y construcción del corpus de pares, en 4 vías que
  comparten una blacklist común para no reprocesar sentencias:
  - `ds_parte1_redal.ipynb` — base curada a mano desde redal.org
  - `ds_parte2_corte_const.ipynb` — muestreo aleatorio sobre el índice completo de tutelas
    T/SU de la Corte Constitucional
  - `ds_parte3_research_list.ipynb` — candidatas sugeridas por investigación externa
    (Claude web)
  - `ds_parte4_sl_casacion.ipynb` — providencias SL de la Sala de Casación Laboral de la
    Corte Suprema (cubre jornada, factores salariales y contratación, temas que la tutela
    subrepresenta estructuralmente)
  - `ds_diccionario_articulos.ipynb` — texto completo de cada norma citada por el corpus
- `notebooks/finetuning/` — notebook de entrega: carga del modelo base, LoRA, entrenamiento,
  baseline y evaluación, ejemplos cualitativos
- `data/` — dataset de entrenamiento y sus insumos (ver `data/README.md` para el detalle de
  cada archivo)
- `docs/` — definición del proyecto (`STAI_Plantilla_Proyecto.md`), descripción del dataset,
  reporte final

## Split del equipo (M1)

- **Pablo** — dataset (`notebooks/dataset/`)
- **Miguel** — training LoRA (`notebooks/finetuning/`)
- **Samuel** — baseline + evaluación (dentro de `notebooks/finetuning/`)
- **Martin** — docs + reporte (`docs/`)

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` y pon tu `ANTHROPIC_API_KEY` (necesaria para los pasos de extracción vía LLM en
`notebooks/dataset/`).
