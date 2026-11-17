"""Sistema final: agente + búsqueda híbrida + reranking + decoder, todo en CPU.

Es el mismo pipeline de M3 (agente de un paso sobre C2), empaquetado como un módulo que
cargan tanto el notebook de eficiencia como la app. El decoder puede ser el de M3
(transformers) o una versión cuantizada en GGUF (llama.cpp); lo demás no cambia.
"""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, ValidationError

RAIZ = Path(__file__).resolve().parents[1]

EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
GEN_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
GGUF_REPO = 'Qwen/Qwen2.5-1.5B-Instruct-GGUF'
GGUF_ARCHIVOS = {
    'fp16': 'qwen2.5-1.5b-instruct-fp16.gguf',
    'q8_0': 'qwen2.5-1.5b-instruct-q8_0.gguf',
    'q4_k_m': 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
}
MODELO_CE = 'dccuchile/bert-base-spanish-wwm-cased'
RUTA_ADAPTADOR = RAIZ / 's04_lora_adapter'
N_CAND = 20
K_RRF = 60
TOP_K = 3
DESC_CHARS = 400
MAX_TOKENS_DECODER = 120
SEED = 42
NL = chr(10)
NOMBRE_TOOL = 'buscar_articulos_laborales'
VERSION = '1.0'

AVISO = ('Este resultado es informativo y no constituye una recomendación jurídica personalizada. '
         'Se basa en un diccionario local de artículos; no verifica hechos ni vigencia.')

SISTEMA_DECODER = (
    'Eres un asistente informativo de derecho laboral colombiano. Recibes una consulta y tres referencias '
    'normativas recuperadas automáticamente, cada una con una descripción. Escribe un resumen neutro de dos o '
    'tres frases que diga qué establecen las referencias en relación con el tema de la consulta. No afirmes que '
    'una norma aplica o no aplica al caso, no agregues normas ni datos que no estén en las descripciones y no '
    'des consejos ni recomendaciones. Si las referencias no tratan el tema de la consulta, dilo.'
)

SISTEMA_AGENTE = """
Eres el controlador de herramientas de un asistente especializado
en derecho laboral individual colombiano.

Tu única tarea es decidir si una pregunta requiere utilizar
la herramienta buscar_articulos_laborales.

No debes responder la pregunta, redactar asesoría jurídica
ni generar explicaciones adicionales.

REGLAS DE DECISIÓN:

1. Utiliza buscar_articulos_laborales cuando la consulta esté
   relacionada con trabajo, empleados, contratos laborales,
   salarios, despidos, incapacidades, discriminación laboral,
   protección laboral o derechos de los trabajadores.

2. Si existen dudas sobre si la consulta pertenece al ámbito
   laboral, utiliza buscar_articulos_laborales.

3. Utiliza no_tool únicamente cuando el asunto sea claramente
   ajeno al derecho laboral, como una herencia familiar
   sin relación con un vínculo de trabajo.

FORMATO OBLIGATORIO DE RESPUESTA:

Debes devolver exclusivamente un objeto JSON válido.

El objeto debe contener exactamente dos propiedades:
- "action"
- "arguments"

La propiedad "action" solo puede contener:
- "buscar_articulos_laborales"
- "no_tool"

La propiedad "arguments" debe ser siempre un objeto JSON.

SI DEBES UTILIZAR LA HERRAMIENTA:

Devuelve exactamente esta estructura:

{
  "action": "buscar_articulos_laborales",
  "arguments": {
    "consulta": "PREGUNTA ORIGINAL COMPLETA DEL USUARIO"
  }
}

Copia la pregunta original completa en arguments.consulta.
No la resumas, reformules, traduzcas ni elimines información.
Conserva los hechos, fechas, circunstancias y términos jurídicos.

SI NO DEBES UTILIZAR LA HERRAMIENTA:

Devuelve exactamente esta estructura:

{
  "action": "no_tool",
  "arguments": {}
}

RESTRICCIONES:

- No escribas texto antes ni después del JSON.
- No utilices bloques Markdown.
- No agregues propiedades adicionales.
- No coloques "consulta" fuera de "arguments".
- No inventes acciones diferentes de las dos permitidas.
- No respondas la pregunta jurídica.

EJEMPLOS DE DECISIÓN:

- Despido durante el embarazo:
  buscar_articulos_laborales.

- Renuncia forzada de una persona con discapacidad:
  buscar_articulos_laborales.

- Repartición de una herencia familiar sin relación laboral:
  no_tool.

Devuelve únicamente el objeto JSON correspondiente
a la pregunta recibida.
""".strip()

SISTEMA_AGENTE_REINTENTO = SISTEMA_AGENTE + """

Tu respuesta anterior no cumplió el contrato.

Debes devolver exactamente una de estas estructuras:

Para buscar:
{"action":"buscar_articulos_laborales",
 "arguments":{"consulta":"PREGUNTA ORIGINAL COMPLETA"}}

Para no buscar:
{"action":"no_tool","arguments":{}}

No pongas consulta fuera de arguments.
No agregues texto explicativo.
"""


# ---------------------------------------------------------------- contrato de salida

@dataclass
class Resultado:
    """Lo que devuelve `Sistema.responder`. Todo es serializable a JSON y no guarda la pregunta."""
    respuesta: str                                   # texto completo que ve el usuario
    referencias: list[dict[str, str]]                # [{"cita", "url"}], hasta 3
    componentes: list[str]                           # de: agente, rag, reranker, decoder
    valvula: bool                                    # True si el agente decidió no buscar
    latencias_ms: dict[str, float] = field(default_factory=dict)
    accion_agente: str = ''                          # buscar_articulos_laborales | no_tool | fallback
    error: Optional[str] = None

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- corpus y texto

def cargar_corpus(raiz: Path = RAIZ) -> pd.DataFrame:
    d = pd.read_csv(raiz / 'data/diccionario_articulos.csv', dtype=str)
    d['cita'] = d['fuente'] + ' Art. ' + d['numero']
    return d.rename(columns={'texto_completo': 'texto', 'url_fuente': 'url'})[['cita', 'fuente', 'texto', 'url']]


_NOTA = re.compile(r'(?:Modificado|Adicionado|Reglamentado|Derogado|Corregido)[^<]{0,200}?\bde\s+\d{4}\s*\.?\s*(?:<[^>]*>\s*)?')
_MARCAS = re.compile(r'\b(?:Jurisprudencia|Concordancias|Legislación Anterior|Notas de Vigencia|Doctrina)\s*:')
_ENCABEZADO = re.compile(r'^\s*(?:ART[IÍ]CULO|Art[íi]culo)\s+\d+[A-Za-z]?[º°o]?\s*[.\-:]*\s*')


def descripcion(texto: str, max_chars: int = DESC_CHARS) -> str:
    t = _ENCABEZADO.sub('', texto)
    t = _NOTA.sub('', t)
    t = re.sub(r'<[^>]*>', '', t)
    t = _MARCAS.split(t)[0]
    t = re.sub(r'\s+', ' ', t).strip()
    if len(t) <= max_chars:
        return t
    corte = t[:max_chars].rsplit(' ', 1)[0]
    return corte.rstrip(' ,;:') + '…'


def referencias_de(indices, corpus: pd.DataFrame) -> list[dict[str, str]]:
    return [{'cita': corpus.at[i, 'cita'], 'texto': corpus.at[i, 'texto'], 'url': corpus.at[i, 'url']} for i in indices]


def prompt_decoder(consulta: str, referencias: list[dict]) -> str:
    lineas = '\n'.join(f"{i}. {r['cita']}: {descripcion(r['texto'])}" for i, r in enumerate(referencias, 1))
    return (f'Consulta del trabajador:\n{consulta}\n\nReferencias recuperadas:\n{lineas}\n\n'
            'Escribe solo el resumen, sin encabezados ni listas.')


def limpiar_resumen(texto: str) -> str:
    t = re.sub(r'^\s*Resumen\s*:\s*', '', texto.strip(), flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', t)[:700]


def armar_respuesta(referencias: list[dict], resumen: str) -> str:
    lineas = NL.join(f"{i}. {r['cita']}: {descripcion(r['texto'])} ({r['url']})" for i, r in enumerate(referencias, 1))
    return f'Referencias recuperadas:{NL}{lineas}{NL}{NL}Resumen: {resumen}{NL}{NL}{AVISO}'


def respuesta_sin_contexto() -> str:
    resumen = ('La consulta está fuera del alcance del corpus local de derecho laboral colombiano; '
               'por eso no se recuperaron artículos para responderla.')
    return armar_respuesta([], resumen)


def tokenizar(texto: str) -> list[str]:
    texto = unicodedata.normalize('NFD', texto.lower())
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    return re.findall(r'\w+', texto)


def fusionar_rrf(rankings, k_rrf: int = K_RRF, top: Optional[int] = None):
    puntos: dict[int, float] = {}
    for ranking in rankings:
        for puesto, (idx, _) in enumerate(ranking):
            puntos[idx] = puntos.get(idx, 0) + 1.0 / (k_rrf + puesto + 1)
    fusion = sorted(puntos.items(), key=lambda x: -x[1])
    return fusion[:top] if top else fusion


# ---------------------------------------------------------------- decisión del agente

class ArgumentosHerramienta(BaseModel):
    consulta: Optional[str] = None
    model_config = ConfigDict(extra='forbid')


class DecisionHerramienta(BaseModel):
    action: Literal['buscar_articulos_laborales', 'no_tool']
    arguments: ArgumentosHerramienta
    model_config = ConfigDict(extra='forbid')


def extraer_primer_json(texto: str):
    if not isinstance(texto, str):
        return None
    limpio = texto.replace('```json', '').replace('```JSON', '').replace('```', '').strip()
    decoder = json.JSONDecoder()
    for posicion, caracter in enumerate(limpio):
        if caracter != '{':
            continue
        try:
            valor, _ = decoder.raw_decode(limpio[posicion:])
            if isinstance(valor, dict):
                return valor
        except json.JSONDecodeError:
            continue
    return None


def parsear_decision(texto: str, pregunta: str) -> dict:
    """Separa la propuesta del modelo de la acción efectiva. Nunca decide por palabras clave."""
    propuesta = extraer_primer_json(texto)
    if propuesta is None:
        return {'action': 'fallback', 'consulta': pregunta, 'motivo': 'json_invalido'}
    try:
        validada = DecisionHerramienta.model_validate(propuesta)
    except ValidationError:
        return {'action': 'fallback', 'consulta': pregunta, 'motivo': 'schema_invalido'}
    if validada.action == 'no_tool':
        return {'action': 'no_tool', 'consulta': None, 'motivo': 'modelo'}
    # Se busca siempre con la pregunta original: la reformulación del modelo puede perder hechos.
    return {'action': NOMBRE_TOOL, 'consulta': pregunta, 'motivo': 'modelo_consulta_original'}


def _cumple_schema(texto: str) -> bool:
    propuesta = extraer_primer_json(texto)
    if propuesta is None:
        return False
    try:
        DecisionHerramienta.model_validate(propuesta)
        return True
    except ValidationError:
        return False


# ---------------------------------------------------------------- el sistema

class Sistema:
    """Pipeline completo en CPU.

    decoder: 'hf' (Qwen con transformers, como en M3) o una clave de GGUF_ARCHIVOS
             ('fp16', 'q8_0', 'q4_k_m') servida con llama.cpp.
    ce_int8: cuantiza dinámicamente a int8 las capas lineales del cross-encoder.
    """

    def __init__(self, decoder: str = 'hf', ce_int8: bool = False, hilos: int = 2, raiz: Path = RAIZ):
        if decoder != 'hf' and decoder not in GGUF_ARCHIVOS:
            raise ValueError(f"decoder debe ser 'hf' o uno de {list(GGUF_ARCHIVOS)}")
        self.decoder, self.ce_int8, self.hilos, self.raiz = decoder, ce_int8, hilos, raiz
        self.cargado = False
        self.historial: list[Resultado] = []
        # interfaz que espera el harness de M2
        self.ultimo_contexto: list[dict] = []
        self.latencia_recuperacion = 0.0
        self.consultas_reformuladas: list[str] = []
        self.ruta_gguf: Optional[str] = None

    @property
    def nombre(self) -> str:
        return f"decoder={self.decoder}" + (' + cross-encoder int8' if self.ce_int8 else '')

    # -- carga (una sola vez, al arrancar) --
    def cargar(self) -> 'Sistema':
        import chromadb
        from peft import PeftModel
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(self.hilos)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        self.corpus = cargar_corpus(self.raiz)
        self.textos_ce = (self.corpus['fuente'] + '. ' + self.corpus['texto']).tolist()

        self.embeddings = SentenceTransformer(EMBED_MODEL, device='cpu')
        self.chroma = chromadb.Client()
        try:
            self.chroma.delete_collection('final')
        except Exception:
            pass
        self.coleccion = self.chroma.create_collection('final', metadata={'hnsw:space': 'cosine'})
        self.coleccion.add(ids=[str(i) for i in range(len(self.corpus))], documents=self.corpus['texto'].tolist(),
                           embeddings=self.embeddings.encode(self.corpus['texto'].tolist(), show_progress_bar=False).tolist())
        self.bm25 = BM25Okapi([tokenizar(t) for t in self.corpus['texto']])

        self.ce_tok = AutoTokenizer.from_pretrained(MODELO_CE)
        base = AutoModelForSequenceClassification.from_pretrained(MODELO_CE, num_labels=2)
        ce = PeftModel.from_pretrained(base, RUTA_ADAPTADOR).eval()
        if self.ce_int8:
            ce = torch.ao.quantization.quantize_dynamic(ce.merge_and_unload().eval(), {torch.nn.Linear}, dtype=torch.qint8)
        self.ce = ce

        self._cargar_decoder()
        self.cargado = True
        return self

    def _cargar_decoder(self) -> None:
        if self.decoder == 'hf':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(GEN_MODEL)
            # float32: en CPU bf16 no tiene kernels rápidos y haría la línea base artificialmente lenta
            self.lm = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.float32).eval()
        else:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama
            self.ruta_gguf = hf_hub_download(GGUF_REPO, GGUF_ARCHIVOS[self.decoder])
            self.lm = Llama(model_path=self.ruta_gguf, n_ctx=2048, n_threads=self.hilos,
                            n_gpu_layers=0, verbose=False, seed=0)

    def tamano_decoder_mb(self) -> float:
        if self.decoder == 'hf':
            return sum(p.numel() * p.element_size() for p in self.lm.parameters()) / 1e6
        return Path(self.ruta_gguf).stat().st_size / 1e6

    def tamano_cross_encoder_mb(self) -> float:
        import io
        buffer = io.BytesIO()
        torch.save(self.ce.state_dict(), buffer)
        return buffer.getbuffer().nbytes / 1e6

    # -- generación --
    def generar(self, sistema: str, usuario: str, max_tokens: int) -> str:
        if self.decoder == 'hf':
            chat = [{'role': 'system', 'content': sistema}, {'role': 'user', 'content': usuario}]
            prompt = self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            entrada = self.tok(prompt, return_tensors='pt')
            with torch.no_grad():
                salida = self.lm.generate(**entrada, max_new_tokens=max_tokens, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
            return self.tok.decode(salida[0][entrada.input_ids.shape[1]:], skip_special_tokens=True).strip()
        o = self.lm.create_chat_completion(
            messages=[{'role': 'system', 'content': sistema}, {'role': 'user', 'content': usuario}],
            temperature=0, max_tokens=max_tokens, seed=0)
        return o['choices'][0]['message']['content'].strip()

    # -- recuperación (C2: híbrida + reranking) --
    def _densa(self, consulta: str, n: int = N_CAND):
        r = self.coleccion.query(query_embeddings=self.embeddings.encode([consulta]).tolist(), n_results=n)
        return [(int(i), 1 - d) for i, d in zip(r['ids'][0], r['distances'][0])]

    def _bm25(self, consulta: str, n: int = N_CAND):
        scores = self.bm25.get_scores(tokenizar(consulta))
        return [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1][:n]]

    def hibrida(self, consulta: str) -> list[int]:
        return [i for i, _ in fusionar_rrf([self._densa(consulta), self._bm25(consulta)])[:N_CAND]]

    def rerank(self, consulta: str, candidatos: list[int]) -> list[int]:
        entradas = self.ce_tok([consulta] * len(candidatos), [self.textos_ce[i] for i in candidatos],
                               truncation='only_second', max_length=512, padding=True, return_tensors='pt')
        with torch.no_grad():
            logits = self.ce(**entradas).logits
        puntajes = torch.softmax(logits, dim=-1)[:, 1].tolist()
        return [i for i, _ in sorted(zip(candidatos, puntajes), key=lambda x: -x[1])[:TOP_K]]

    # -- agente --
    def decidir(self, pregunta: str) -> dict:
        propuesta = self.generar(SISTEMA_AGENTE, f'Pregunta del usuario:\n{pregunta}', 96)
        intentos = 1
        if not _cumple_schema(propuesta):
            propuesta = self.generar(SISTEMA_AGENTE_REINTENTO, f'Pregunta del usuario:\n{pregunta}', 96)
            intentos = 2
        decision = parsear_decision(propuesta, pregunta)
        decision['intentos_formato'] = intentos
        return decision

    # -- punto de entrada --
    def responder(self, pregunta: str) -> Resultado:
        if not self.cargado:
            raise RuntimeError('Llame a cargar() antes de responder.')
        lat: dict[str, float] = {}
        comps: list[str] = []
        t_total = time.perf_counter()
        try:
            t = time.perf_counter()
            decision = self.decidir(pregunta)
            lat['agente_ms'] = (time.perf_counter() - t) * 1000
            comps.append('agente')
            if decision['action'] == 'no_tool':
                self.ultimo_contexto, self.latencia_recuperacion = [], 0.0
                res = Resultado(respuesta_sin_contexto(), [], comps, True, accion_agente='no_tool')
            else:
                t = time.perf_counter()
                candidatos = self.hibrida(decision['consulta'])
                lat['hibrida_ms'] = (time.perf_counter() - t) * 1000
                t = time.perf_counter()
                indices = self.rerank(decision['consulta'], candidatos)
                lat['reranker_ms'] = (time.perf_counter() - t) * 1000
                self.latencia_recuperacion = (lat['hibrida_ms'] + lat['reranker_ms']) / 1000
                comps += ['rag', 'reranker']
                self.ultimo_contexto = referencias_de(indices, self.corpus)
                t = time.perf_counter()
                salida = self.generar(SISTEMA_DECODER, prompt_decoder(pregunta, self.ultimo_contexto), MAX_TOKENS_DECODER)
                lat['decoder_ms'] = (time.perf_counter() - t) * 1000
                comps.append('decoder')
                texto = armar_respuesta(self.ultimo_contexto, limpiar_resumen(salida) or 'No se obtuvo resumen.')
                res = Resultado(texto, [{'cita': r['cita'], 'url': r['url']} for r in self.ultimo_contexto],
                                comps, False, accion_agente=decision['action'])
        except Exception as exc:  # el servicio no debe caerse por una petición
            self.ultimo_contexto, self.latencia_recuperacion = [], 0.0
            res = Resultado('Ocurrió un error al procesar la consulta.', [], comps, False,
                            accion_agente='error', error=type(exc).__name__)
        lat['total_ms'] = (time.perf_counter() - t_total) * 1000
        res.latencias_ms = {k: round(v, 1) for k, v in lat.items()}
        self.historial.append(res)
        return res

    def __call__(self, pregunta: str) -> str:
        """Interfaz del harness de M2: devuelve el texto y deja `ultimo_contexto` listo."""
        return self.responder(pregunta).respuesta

    # -- health check: no genera texto ni hace una consulta real --
    def salud(self) -> dict[str, Any]:
        if not self.cargado:
            return {'status': 'cargando', 'version': VERSION}
        indice_ok = self.coleccion.count() == len(self.corpus) and len(self.bm25.get_scores(['trabajo'])) == len(self.corpus)
        return {'status': 'ok' if indice_ok else 'degradado', 'version': VERSION, 'decoder': self.decoder,
                'cross_encoder_int8': self.ce_int8, 'articulos_indexados': int(self.coleccion.count())}
