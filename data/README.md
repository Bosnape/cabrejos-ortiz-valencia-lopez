# Qué es cada archivo en esta carpeta (data/)

Estado del corpus: **138 sentencias / 561 pares**. Se dejó de ampliar deliberadamente en este
punto.

## Dataset final

- **`dataset_cross_encoder.csv`** — el dataset de entrenamiento. Columnas: `consulta`
  (situación en lenguaje coloquial), `articulo` (norma citada), `tipo`
  (`positivo`/`negativo_facil`/`negativo_dificil_placeholder`), `label` (1/0),
  `sentencia_origen`.
- **`diccionario_articulos.csv`** — texto completo de cada artículo citado en el dataset.
  Columnas: `fuente`, `numero`, `texto_completo`, `n_citas_en_dataset`, `url_fuente`. Se une a
  `dataset_cross_encoder.csv` por `(fuente, numero)` — ver la sección "Diccionario de
  artículos" más abajo para el diseño completo. Se regenera corriendo
  `notebooks/dataset/ds_diccionario_articulos.ipynb`.

## Insumos del pipeline (no derivables — no borrar)

- **`descartados.csv`** — blacklist compartida por los 4 notebooks de `notebooks/dataset/`
  (columnas `sentencia`, `razon`). Evita que una sentencia ya vista/rechazada se reprocese.
  Necesario para que los notebooks sean incrementales.
- **`candidatas_research.csv`** — sentencias sugeridas por investigación externa (Claude web),
  consumido por `ds_parte3_research_list.ipynb` (fuentes tipo Corte Constitucional, HTML).
- **`fuentes_normas.csv`** — URL real de cada una de las ~40 normas (Leyes/Decretos/CST/CP)
  citadas en el dataset, resueltas a mano vía búsqueda (no tienen un patrón de URL
  predecible). Insumo de `ds_diccionario_articulos.ipynb`. **No es derivable por código** — si
  se pierde, hay que rehacer la investigación.
- **`SL2600-2025.pdf`** — PDF de ejemplo usado en `decisiones_base_encoder.ipynb`, para
  reproducibilidad.

---

## Diccionario de artículos — diseño

El texto completo de cada artículo/ley vive separado del dataset de pares (en
`diccionario_articulos.csv`), no embebido en `dataset_cross_encoder.csv`. Este último solo
guarda el identificador (`"CST Art. 127"`).

**Por qué:**
1. **Evita duplicar texto.** Un mismo artículo se cita en varias filas (ej. `CST Art. 62`
   aparece ~15 veces) — si el texto completo viviera en cada fila, se repetiría.
2. **Se necesita también en producción, no solo para entrenar.** El asistente rankea
   artículos candidatos contra una consulta nueva en tiempo real — necesita el mismo
   diccionario para eso.
3. **Un solo lugar para corregir.** Si el texto de un artículo tiene un error, se arregla en
   una fila del diccionario, no en cada fila del dataset que lo cita.

**Cómo se usa** — join por `(fuente, numero)` antes de tokenizar, tanto al entrenar como al
servir el modelo:

```python
df_pares = pd.read_csv("data/dataset_cross_encoder.csv")
df_dic = pd.read_csv("data/diccionario_articulos.csv")
# "articulo" (ej. "CST Art. 127") se normaliza a (fuente, numero) -- ver
# normalizar_articulos_citados() en ds_diccionario_articulos.ipynb -- y se une con df_dic
```

**Cobertura actual: 143/145 artículos (98.6%)**, merge real contra el dataset: 95.7% de las
filas con texto completo. Lo que falta:
- **2 artículos sin fuente pública posible** — convenciones colectivas privadas entre una
  empresa y un sindicato específico, no codificadas en ningún repositorio público. Se quedan
  fuera para siempre, no por falta de esfuerzo.
- **22 filas del dataset citan una ley/decreto completo sin número de artículo específico**
  (ej. `"Ley 361 de 1997"`) — decisión deliberada, no participan del join por artículo.
