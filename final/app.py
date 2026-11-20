"""App del sistema final: interfaz web, health check, log JSON por petición y métricas en MLflow.

Cómo correrla (desde la raíz del repositorio, con `pip install -r final/requirements.txt`):

    python final/app.py                    # http://localhost:7860, /health responde aunque el modelo siga cargando

Variables de entorno (todas opcionales):
    DISPOSITIVO     cpu | cuda. Por defecto cuda si hay GPU.
    DECODER         hf | hf_int8 | hf_nf4 | fp16 | q8_0 | q4_k_m. Por defecto hf con GPU y q8_0 (GGUF) con CPU.
    PORT            puerto, 7860 por defecto.
    MLFLOW_TRACKING_URI  si existe, cada petición se registra como métricas en MLflow (p. ej. sqlite:///mlflow.db).
    MLFLOW_EXPERIMENT    nombre del experimento, "asistente-legal-laboral" por defecto.
    LOG_PETICIONES  ruta del archivo de log, "peticiones.jsonl" por defecto.

Privacidad: el log y MLflow guardan métricas de la petición (hora, latencias, componentes, válvula, error). No guardan
el texto de la consulta ni de la respuesta; de la consulta solo se registra su longitud.
"""
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sistema as S  # noqa: E402

VERSION = S.VERSION
T0 = time.time()

try:
    import torch
    DISPOSITIVO = os.environ.get('DISPOSITIVO') or ('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    DISPOSITIVO = os.environ.get('DISPOSITIVO', 'cpu')
DECODER = os.environ.get('DECODER') or ('hf' if DISPOSITIVO == 'cuda' else 'q8_0')

# ---- una línea JSON por petición, con un logger propio (basicConfig mezclaría mensajes de otras librerías) ----
log = logging.getLogger('peticiones')
log.setLevel(logging.INFO)
log.propagate = False
for h in (logging.FileHandler(os.environ.get('LOG_PETICIONES', 'peticiones.jsonl'), encoding='utf-8'),
          logging.StreamHandler()):
    h.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(h)

# ---- MLflow solo si se indica dónde guardar ----
mlflow = None
if os.environ.get('MLFLOW_TRACKING_URI'):
    import mlflow
    mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT', 'asistente-legal-laboral'))
    # El run activo de MLflow es local a cada hilo y Gradio atiende en otros hilos: se escribe siempre por run_id.
    MLFLOW_RUN_ID = mlflow.start_run(run_name=f'app-v{VERSION}-{DISPOSITIVO}').info.run_id
    mlflow.log_params({'version': VERSION, 'dispositivo': DISPOSITIVO, 'decoder': DECODER})

# ---- el sistema se carga UNA vez, en segundo plano, para que /health responda mientras tanto ----
SISTEMA: S.Sistema | None = None
ERROR_CARGA: str | None = None
CANDADO = threading.Lock()          # un solo modelo atiende una petición a la vez
CONTADOR = {'peticiones': 0, 'valvula': 0, 'errores': 0}


def _cargar() -> None:
    global SISTEMA, ERROR_CARGA
    try:
        SISTEMA = S.Sistema(decoder=DECODER, dispositivo=DISPOSITIVO).cargar()
    except Exception as exc:  # el servicio sigue vivo y /health lo dice
        ERROR_CARGA = type(exc).__name__


threading.Thread(target=_cargar, daemon=True).start()

AVISO_UI = (
    'Asistente **informativo** de derecho laboral individual colombiano. Muestra los artículos que más se relacionan con '
    'tu consulta y un resumen; **no es una recomendación jurídica** y puede equivocarse. Verifica siempre las citas. '
    'No escribas datos personales (nombres, cédulas, empresas). El sistema no guarda el texto de tu consulta.'
)


def responder(pregunta: str) -> str:
    rid = uuid.uuid4().hex[:8]
    pregunta = (pregunta or '').strip()
    if not pregunta:
        return 'Escribe una consulta.'
    if SISTEMA is None:
        return ('El sistema todavía está cargando los modelos.' if ERROR_CARGA is None
                else 'El sistema no pudo iniciar. Revisa /health.')
    with CANDADO:
        res = SISTEMA.responder(pregunta)
    CONTADOR['peticiones'] += 1
    CONTADOR['valvula'] += int(res.valvula)
    CONTADOR['errores'] += int(res.error is not None)
    registro = {
        'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'id': rid, 'version': VERSION,
        'dispositivo': DISPOSITIVO, 'decoder': DECODER,
        'componentes': res.componentes, 'accion_agente': res.accion_agente, 'valvula': res.valvula,
        'n_referencias': len(res.referencias), 'longitud_pregunta': len(pregunta),
        'latencia_ms': res.latencias_ms.get('total_ms'), 'latencias_ms': res.latencias_ms, 'error': res.error,
    }
    log.info(json.dumps(registro, ensure_ascii=False))
    if mlflow:
        try:
            metricas = {'latencia_ms': registro['latencia_ms'], 'valvula': int(res.valvula),
                        'error': int(res.error is not None), 'peticiones_total': CONTADOR['peticiones'],
                        **{k: v for k, v in res.latencias_ms.items() if k != 'total_ms'}}
            mlflow.log_metrics({k: v for k, v in metricas.items() if v is not None},
                               step=CONTADOR['peticiones'], run_id=MLFLOW_RUN_ID)
        except Exception:  # la observabilidad nunca debe tumbar la respuesta
            pass
    return res.respuesta


app = FastAPI(title='Asistente de derecho laboral')


@app.get('/health')
def health():
    """Vivo sin hacer una pregunta real: modelos cargados e índice respondiendo (no genera texto)."""
    base = {'version': VERSION, 'dispositivo': DISPOSITIVO, 'decoder': DECODER, 'uptime_s': int(time.time() - T0),
            'mlflow': 'activo' if mlflow else 'inactivo', **CONTADOR}
    if ERROR_CARGA:
        return {'status': 'error', 'error_carga': ERROR_CARGA, **base}
    if SISTEMA is None:
        return {'status': 'cargando', **base}
    return {**SISTEMA.salud(), **base}


with gr.Blocks(title='Asistente de derecho laboral') as demo:
    gr.Markdown('# Asistente de consulta de derecho laboral\n' + AVISO_UI)
    entrada = gr.Textbox(label='Describe tu consulta', lines=6)
    salida = gr.Textbox(label='Respuesta', lines=14)
    gr.Button('Consultar', variant='primary').click(responder, entrada, salida, api_name='responder')

demo.queue(default_concurrency_limit=1)
app = gr.mount_gradio_app(app, demo, path='/')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
