"""Inspeccion estatica de notebooks M3; no ejecuta celdas, modelos ni llamadas API."""

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "notebooks/evaluation/harness_de_evaluacion.ipynb",
    "notebooks/rag/rag_ingenuo.ipynb",
    "notebooks/rag/rag_avanzado.ipynb",
    "notebooks/rag/agente_herramientas.ipynb",
    "notebooks/rag/ragas_m3.ipynb",
]


def git(*args):
    return subprocess.run(["git", "-C", str(RAIZ), *args], capture_output=True, check=True).stdout


def inspeccionar(ruta, leer):
    raw = leer(ruta)
    if raw is None:
        return {"archivo": ruta, "estado": "pendiente: archivo ausente"}
    nb = json.loads(raw)
    celdas = [(i, c) for i, c in enumerate(nb["cells"], 1) if c["cell_type"] == "code"]
    codigo = "\n".join("".join(c.get("source", [])) for _, c in celdas)
    errores_sintaxis = []
    arboles = []
    errores_guardados = []
    for indice, celda in celdas:
        try:
            arboles.append(ast.parse("".join(celda.get("source", []))))
        except SyntaxError as error:
            errores_sintaxis.append({"celda": indice, "linea": error.lineno, "mensaje": error.msg})
        for salida in celda.get("outputs", []):
            if salida.get("output_type") == "error":
                errores_guardados.append({"celda": indice, "tipo": salida.get("ename")})
    constantes = {}
    for arbol in arboles:
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Assign) and isinstance(nodo.value, ast.Constant):
                for destino in nodo.targets:
                    if isinstance(destino, ast.Name) and destino.id in {
                        "SEED", "EMBED_MODEL", "GEN_MODEL", "JUEZ_MODEL", "MODELO_JUEZ", "TOP_K", "REPO_BRANCH"
                    }:
                        constantes[destino.id] = nodo.value.value
    requeridos = []
    for arbol in arboles:
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "requeridos" for t in nodo.targets):
                if isinstance(nodo.value, ast.List):
                    for item in nodo.value.elts:
                        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "Path":
                            if item.args and isinstance(item.args[0], ast.Constant) and isinstance(item.args[0].value, str):
                                requeridos.append(item.args[0].value)
    return {
        "archivo": ruta,
        "estado": "inspeccionado, sin ejecucion nueva",
        "sha256_texto_utf8_lf": hashlib.sha256(raw.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")).hexdigest(),
        "celdas_codigo": len(celdas),
        "celdas_con_contador_guardado": sum(c.get("execution_count") is not None for _, c in celdas),
        "errores_sintaxis": errores_sintaxis,
        "errores_en_salidas_guardadas": errores_guardados,
        "constantes": constantes,
        "llamadas_semilla_detectadas": [
            llamada for llamada in ["random.seed(SEED)", "np.random.seed(SEED)", "torch.manual_seed(SEED)"]
            if llamada in codigo
        ],
        "archivos_requeridos_en_bootstrap": {p: leer(p) is not None for p in requeridos},
        "colab_limpio": "pendiente: este script no ejecuta notebooks",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="Commit o rama local a inspeccionar sin cambiar de rama; por defecto, archivos locales.")
    args = parser.parse_args()
    # Se resuelve primero a un SHA; git show nunca recibe una opcion arbitraria del usuario.
    revision = git("rev-parse", "--verify", "--end-of-options", (args.ref or "HEAD") + "^{commit}").decode().strip()

    def leer(ruta):
        if args.ref:
            try:
                return git("show", f"{revision}:{ruta}")
            except subprocess.CalledProcessError:
                return None
        p = RAIZ / ruta
        return p.read_bytes() if p.is_file() else None

    versionados = git("ls-tree", "-r", "--name-only", revision).decode().splitlines()
    hallazgos_secretos = []
    patrones = {
        "clave_anthropic": rb"sk-ant-[A-Za-z0-9_-]{20,}",
        "token_github": rb"gh[pousr]_[A-Za-z0-9]{30,}",
        "clave_privada": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    }
    inspeccionados = 0
    for ruta in versionados:
        if Path(ruta).suffix in {".md", ".json", ".ipynb", ".csv", ".py", ".txt", ".example"} or ruta == ".env":
            contenido = leer(ruta)
            if contenido is not None:
                inspeccionados += 1
                for tipo, patron in patrones.items():
                    if re.search(patron, contenido):
                        hallazgos_secretos.append({"archivo": ruta, "tipo": tipo})
    casos = json.loads(leer("notebooks/evaluation/eval_set.json"))
    resultado = {
        "revision_base": revision,
        "fuente": args.ref or "archivos locales; puede incluir cambios sin commit",
        "alcance": "JSON, sintaxis Python, semillas declaradas, archivos requeridos y salidas guardadas. No prueba ejecucion.",
        "controles_repositorio": {
            "requirements_versionados": [p for p in versionados if Path(p).name == "requirements.txt"],
            "env_versionado": ".env" in versionados,
            "env_example_presente": leer(".env.example") is not None,
            "gitignore_excluye_env": ".env" in leer(".gitignore").decode().splitlines(),
            "casos_eval_set": len(casos),
            "casos_abstencion": [c["id"] for c in casos if c["case_type"] in {"premisa_falsa", "fuera_de_dominio"}],
            "escaneo_secretos": {
                "archivos_texto_versionados_revisados": inspeccionados,
                "patrones": list(patrones),
                "hallazgos": hallazgos_secretos,
                "limite": "Patrones conocidos; no garantiza ausencia de otros secretos ni revisa todo el historial.",
            },
        },
        "notebooks": [inspeccionar(p, leer) for p in NOTEBOOKS],
    }
    print(json.dumps(resultado, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
