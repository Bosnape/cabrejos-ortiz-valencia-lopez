# Técnica de eficiencia: cuantizar el decoder

Código: `final/sistema.py` (el sistema) y `final/eficiencia/medir_eficiencia.ipynb` (la medición). Resultados completos en
`final/eficiencia/resultados/gpu/` y `final/eficiencia/resultados/cpu/`.

## Qué se cuantizó y por qué

El componente es **Qwen2.5-1.5B-Instruct**, que en el sistema hace dos cosas: decide si el agente busca o no, y escribe el
resumen. Es el que más memoria ocupa (3087 MB, contra 443 MB del cross-encoder) y el que más tarda. En la T4 la latencia
mediana de M3 se reparte así:

| Componente | Mediana |
|---|---|
| Agente (decoder) | 3.16 s |
| Recuperación (híbrida + reranking) | 0.64 s |
| Resumen (decoder) | 6.10 s |

Es decir, el decoder (agente y resumen) es casi toda la latencia: unos 9.3 s de 9.6 s. En la CPU de Colab (2 hilos) la recuperación pesa mucho más
(unos 32 s de 177 s), pero el decoder sigue siendo la mayor parte.

## Umbral, escrito antes de medir

> Aceptamos perder hasta **1 acierto de 13** en el harness, sin bajar el recall@3 medio ni los aciertos en los casos
> adversariales (en particular los de abstención, M2-11 y M2-12), si la latencia **mediana** baja de **5 s**.

## Cómo se midió

Mismo eval set de 13 casos, mismo harness y mismo juez (Claude Haiku, rúbrica 2.0.0) que M2 y M3. Cada versión se mide
en la misma sesión y la misma máquina que su línea base. Se corrió en dos entornos porque el despliegue puede acabar en
cualquiera de los dos:

- **GPU (T4 de Colab)**, la misma máquina de las filas de M2 y M3. Cuantización con bitsandbytes (8 y 4 bits).
- **CPU (2 hilos de Colab)**, como un servidor gratuito. Cuantización GGUF con llama.cpp. bitsandbytes necesita una GPU
  NVIDIA y el GGUF de este experimento es solo para CPU, por eso la técnica se implementa distinto en cada entorno.

## Resultados en GPU (T4)

| Versión | Aciertos | Recall@3 | Juez medio | Adversariales | Agente (de 20) | p50 | p95 | Decoder | VRAM pico |
|---|---|---|---|---|---|---|---|---|---|
| **Antes** (M3, pesos originales) | 3/13 | 0.182 | 2.42 | 3/3 | 18 | 9.55 s | 12.34 s | 3087 MB | 8365 MB |
| bitsandbytes int8 | 1/13 | 0.182 | 2.15 | 1/3 | 19 | 29.08 s | 34.86 s | 1777 MB | 7076 MB |
| bitsandbytes nf4 (4 bits) | 1/13 | 0.182 | 1.88 | 1/3 | 9 | 9.84 s | 14.32 s | 1122 MB | 6519 MB |

- **int8** reduce el decoder a 0.58 veces su tamaño, pero es **3 veces más lenta** (29.1 s contra 9.6 s): en la T4,
  bitsandbytes convierte los datos en cada capa. Pierde dos aciertos, M2-11 y M2-12, los dos casos de abstención.
- **nf4** reduce el decoder a 0.36 veces, pero **no acelera** (9.84 s contra 9.55 s) y daña la válvula: el agente deja de
  producir un JSON válido en 11 de 20 consultas, y solo 8 decisiones coinciden con las de M3. Como el sistema busca por
  defecto cuando el JSON es inválido, el recall no cambia, pero el agente deja de poder decir «no busco». También pierde
  M2-11 y M2-12 (el juez les da 1.0).

## Resultados en CPU (2 hilos)

| Versión | Aciertos | Recall@3 | Juez medio | Adversariales | Agente (de 20) | p50 | p95 | Decoder |
|---|---|---|---|---|---|---|---|---|
| **Antes** (transformers, fp32) | 2/13 | 0.182 | 2.23 | 2/3 | 18 | 177.2 s | 257.6 s | 6175 MB |
| GGUF fp16 | 2/13 | 0.182 | 2.04 | 2/3 | 18 | 193.6 s | 246.9 s | 3560 MB |
| **GGUF int8 (q8_0)** | 2/13 | 0.182 | 2.12 | 2/3 | 18 | 144.7 s | 188.9 s | 1895 MB |
| GGUF int4 (q4_k_m) | 1/13 | 0.091 | 2.15 | 1/3 | 15 | 110.8 s | 152.0 s | 1117 MB |
| GGUF int4 + cross-encoder int8 | 1/13 | 0.091 | 2.12 | 1/3 | 15 | 106.6 s | 139.8 s | 1117 MB |

- **q8_0** conserva la calidad en todas las medidas (mismos aciertos, recall, adversariales y las mismas 20 decisiones del
  agente que M3), con el decoder 3.3 veces más pequeño y la mediana un 18 % más baja.
- **q4_k_m** es más rápida (un 37 % menos) pero pierde la mitad del recall, un adversarial y deja 5 consultas laborales
  sin buscar (contra 2 en M3). El caso que se pierde en el eval set es M2-13, porque el agente decide no buscar.
- La CPU es muy lenta para este sistema: dos minutos por consulta, contra unos 10 s en la T4.

## Pruebas que no miden exactitud

- **Válvula** (20 consultas, 10 laborales y 10 ajenas, solo decide el agente): ver la columna «Agente» de las tablas y los
  CSV `*_decisiones_agente.csv`. El agente de nf4 y el de q4_k_m son los que más se degradan.
- **Casos adversariales** del eval set (M2-11, M2-12 y M2-13): ver la columna «Adversariales».
- **Prueba de invarianza del Módulo 4:** no aplica. Es una prueba sobre la imagen y este sistema no tiene componente
  visual (ver `reporte.md`).

## Veredicto

El umbral **no se cumplió con ninguna versión**.

| Entorno | Versión | Pérdida ≤ 1 acierto | Recall no baja | Adversariales no bajan | p50 < 5 s | ¿Cumple? |
|---|---|---|---|---|---|---|
| GPU | int8 | no (−2) | sí | no | no | no |
| GPU | nf4 | no (−2) | sí | no | no | no |
| CPU | q8_0 | sí | sí | sí | no | no |
| CPU | q4_k_m | sí (−1) | no | no | no | no |

La línea base de M3 en la T4 (9.55 s) ya estaba por encima de los 5 s, de modo que esa condición no era alcanzable con
solo cuantizar el decoder.

## Decisión

- **En GPU, la técnica no sirvió.** Cuantizar con bitsandbytes redujo la memoria pero no la latencia y costó los casos de
  abstención (y, con 4 bits, el JSON del agente). El sistema desplegado en GPU usa los pesos originales de M3.
- **En CPU, la técnica sí sirvió con 8 bits.** `gguf_q8_0` es la versión eficiente si el despliegue tiene que ser en CPU:
  misma calidad, un 18 % menos de latencia y un decoder 3.3 veces más pequeño. Con 4 bits no.

## Cautelas

- Son 13 casos y una sola corrida por versión. Una diferencia de uno o dos casos es ruido, y M2-11 y M2-12 están cerca
  del umbral del juez (≥ 4).
- El juez varía cerca de 0.1 entre corridas con las mismas entradas. Por eso la línea base de esta sesión en la T4 da
  3/13 aciertos y el README de M3 reporta 2/13: en M2-11 el juez dio exactamente 4.0. Las comparaciones válidas son
  entre filas de la misma sesión.
- Con 13 muestras el p95 es casi el máximo observado.
- `gguf_fp16` salió más lento que transformers porque en CPU fp16 no tiene kernels rápidos.
- La memoria en CPU (`rss_carga_mb_aprox` en los CSV) es aproximada, y en GPU no significa nada; lo fiable es el tamaño
  del decoder y la VRAM pico.
