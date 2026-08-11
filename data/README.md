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

**Cobertura actual: 143/145 artículos (98.6%)**, merge real contra el dataset: 95.7% de las
filas con texto completo. Lo que falta:
- **2 artículos sin fuente pública posible** — convenciones colectivas privadas entre una
  empresa y un sindicato específico, no codificadas en ningún repositorio público. Se quedan
  fuera para siempre, no por falta de esfuerzo.
- **22 filas del dataset citan una ley/decreto completo sin número de artículo específico**
  (ej. `"Ley 361 de 1997"`) — decisión deliberada, no participan del join por artículo.

### Cómo se usa: de la fila del dataset al input del modelo

Join por `(fuente, numero)` antes de tokenizar, tanto al entrenar como al servir el modelo.
**Ejemplo ficticio**, paso a paso:

**1. Fila en `dataset_cross_encoder.csv`** — solo tiene el identificador corto:

| consulta | articulo | tipo | label |
|---|---|---|---|
| "Me despidieron sin justa causa mientras estaba incapacitado." | `CST Art. 62` | positivo | 1 |

**2.** `parsear_articulo("CST Art. 62")` lo separa en `("CST", "62")` — esa es la llave (misma
lógica que `normalizar_articulos_citados()` en `ds_diccionario_articulos.ipynb`, aquí en su
forma por-fila).

**3. Fila correspondiente en `diccionario_articulos.csv`**, encontrada por esa llave:

| fuente | numero | texto_completo |
|---|---|---|
| `CST` | `62` | "ARTICULO 62. Son justas causas para dar por terminado unilateralmente el contrato de trabajo: ..." |

`texto_completo` ya trae el número del artículo en su propio encabezado (así salieron los 143
extraídos), pero no dice de qué norma es — por eso el siguiente paso antepone la fuente, no el
número.

**4. Texto final que ve el modelo** (se calcula al momento, no existe como columna en ningún
CSV):

```python
texto_input = fuente + ". " + texto_completo
# "CST. ARTICULO 62. Son justas causas para dar por terminado unilateralmente el contrato..."

tokenizer(
    text=consulta,          # tal cual, sin tocar
    text_pair=texto_input,
    truncation="only_second",
)
# label = 1 (columna `label` de la fila original)
```

Dos cosas a tener en cuenta al hacer esto sobre las 561 filas reales:
- **Filas sin `texto_completo`** (las 24 que no resuelven — sección de cobertura arriba) no
  sirven para el cross-encoder, no hay texto que pasarle como `text_pair`. Hay que dropearlas del
  set de entrenamiento. Son todas `tipo=positivo`, así que el balance final queda 261 positivos /
  276 negativos (no es un drop parejo entre clases, pero sigue balanceado).
- **Truncation:** BETO tiene un límite duro de 512 posiciones — no puede procesar una secuencia
  más larga, sin importar qué tan bien se quisiera conservar todo el texto. Contando con el
  tokenizer real (`dccuchile/bert-base-spanish-wwm-cased`), **124/537 filas (23%)** superan los
  512 tokens al combinar consulta + artículo — algunos artículos llegan a ~1800 tokens. La
  consulta sola nunca pasa de ~135 tokens, así que `truncation="only_second"` (recorta solo el
  artículo) evita que se pierda contexto de la consulta — el default (`"longest_first"`) puede
  empezar a cortarla también una vez se pasa el límite.
