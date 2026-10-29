"""Cuenta los insumos de M3 sin modificarlos; solo requiere Python estandar."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


RAIZ = Path(__file__).resolve().parents[1]


def leer_csv(nombre, columnas):
    ruta = RAIZ / "data" / nombre
    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if not set(columnas).issubset(lector.fieldnames or []):
            raise ValueError(f"Columnas requeridas ausentes en {nombre}: {columnas}")
        filas = list(lector)
    if not filas or any(not fila[c].strip() for fila in filas for c in columnas):
        raise ValueError(f"Archivo vacio o campos requeridos vacios: {nombre}")
    return filas


def resumir():
    articulos = leer_csv("diccionario_articulos.csv", ["fuente", "numero", "texto_completo", "url_fuente"])
    fuentes = leer_csv("fuentes_normas.csv", ["fuente", "url"])
    pares = leer_csv("dataset_cross_encoder.csv", ["sentencia_origen", "consulta", "articulo", "label"])
    llaves = {(f["fuente"], f["numero"]) for f in articulos}
    if len(llaves) != len(articulos):
        raise ValueError("Hay articulos duplicados por (fuente, numero).")
    normas = {f["fuente"] for f in articulos}
    if normas - {f["fuente"] for f in fuentes}:
        raise ValueError("Hay normas del diccionario sin registro en fuentes_normas.csv.")
    dominios = Counter(urlparse(f["url_fuente"]).hostname for f in articulos)
    if None in dominios:
        raise ValueError("Hay una URL de articulo sin dominio.")
    return {
        "normas_origen": len(normas),
        "urls_origen_distintas": len({f["url_fuente"] for f in articulos}),
        "documentos_articulo": len(articulos),
        "chunks_previstos_por_indexacion_actual": len(articulos),
        "unidad_de_chunk": "Una fila/articulo completo, sin ventanas ni solapamiento entre ventanas",
        "sentencias_entrenamiento": len({f["sentencia_origen"] for f in pares}),
        "pares_entrenamiento_en_csv": len(pares),
        "registros_fuentes": len(fuentes),
        "articulos_por_dominio": dict(sorted(dominios.items())),
        "sha256_texto_utf8_lf": {
            f"data/{nombre}": hashlib.sha256(
                (RAIZ / "data" / nombre).read_text(encoding="utf-8-sig").encode("utf-8")
            ).hexdigest()
            for nombre in ["diccionario_articulos.csv", "fuentes_normas.csv", "dataset_cross_encoder.csv"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emite conteos y huellas de los CSV como JSON.")
    args = parser.parse_args()
    datos = resumir()
    if args.json:
        print(json.dumps(datos, ensure_ascii=True, indent=2))
        return
    print("| Medida | Cantidad | Unidad |")
    print("|---|---:|---|")
    for nombre, llave, unidad in [
        ("Normas de origen", "normas_origen", "Fuentes distintas del diccionario"),
        ("Documentos de recuperacion", "documentos_articulo", "Articulos completos"),
        ("Chunks previstos", "chunks_previstos_por_indexacion_actual", "Un articulo por chunk"),
        ("Sentencias de entrenamiento", "sentencias_entrenamiento", "Identificadores de sentencia distintos"),
        ("Pares de entrenamiento", "pares_entrenamiento_en_csv", "Filas del dataset antes de adecuacion"),
    ]:
        print(f"| {nombre} | {datos[llave]} | {unidad} |")
    print("\nDominios de los articulos: " + ", ".join(datos["articulos_por_dominio"]))
    print("Conteo de chunks derivado del codigo actual; no ejecuta ni reconstruye Chroma.")


if __name__ == "__main__":
    main()
