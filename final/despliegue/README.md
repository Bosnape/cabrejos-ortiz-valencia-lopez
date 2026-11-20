# Despliegue en Google Colab

El sistema final se ejecuta en una GPU T4 de Google Colab y se publica temporalmente mediante túneles de Cloudflare. Se abren dos direcciones públicas: una para la aplicación y otra para el panel de MLflow. El procedimiento completo está automatizado en [`despliegue_en_colab.ipynb`](despliegue_en_colab.ipynb).

Se eligió esta alternativa porque el sistema completo necesita GPU para responder en un tiempo razonable. En CPU una consulta tarda alrededor de dos minutos, mientras que la disponibilidad y los créditos de GPU de Hugging Face Spaces no permitieron mantener el despliegue ni hacer comprobaciones de errores. MLflow reemplaza a W&B para registrar la observabilidad de las peticiones.

## Requisitos

- Una cuenta de Google con acceso a Google Colab.
- Un entorno de ejecución con GPU T4.
- Mantener abierta la sesión de Colab durante la demostración.
- No se necesitan tokens, claves ni cuentas de Cloudflare o MLflow.

## Paso a paso

### 1. Abrir el notebook

Abrir `final/despliegue/despliegue_en_colab.ipynb` en Google Colab. El notebook clona la rama principal del repositorio, por lo que los cambios del despliegue deben estar integrados en `main` antes de usarlo para la entrega.

### 2. Seleccionar la GPU

En Colab, ir a **Entorno de ejecución > Cambiar tipo de entorno de ejecución** y seleccionar una GPU T4. La primera celda comprueba que CUDA esté disponible y se detiene con un mensaje claro si el entorno no tiene GPU.

### 3. Preparar el entorno

Ejecutar la primera celda de código. Esta celda:

1. Comprueba y muestra el nombre de la GPU.
2. Clona el repositorio o actualiza la copia existente con `git pull --ff-only`.
3. Elimina una versión antigua de `torchao` cuando es incompatible con PEFT.
4. Instala `final/requirements.txt` conservando la versión de Pillow que trae Colab.

La celda termina mostrando `Listo.`.

### 4. Iniciar MLflow y la aplicación

Ejecutar la celda de la sección **Arrancar la app**. Primero inicia el servidor local de MLflow en el puerto `5000`, usando `mlflow.db` como almacenamiento. Después inicia `final/app.py` en el puerto `7860`.

La aplicación carga los modelos en segundo plano. Durante la carga, `GET /health` responde con `status: cargando`. La celda espera hasta seis minutos y continúa únicamente cuando el estado es `ok`. Si algún proceso falla, consultar:

- `mlflow.log` para errores del servidor de MLflow.
- `app.log` para errores de carga o ejecución de la aplicación.

### 5. Crear las URL públicas

Ejecutar la celda de la sección **Abrir la app y el panel al exterior**. Esta descarga `cloudflared` y abre dos túneles rápidos:

- Aplicación: puerto `7860`.
- Panel de MLflow: puerto `5000`.

La celda imprime tres direcciones:

- `URL de la demo`: interfaz de Gradio.
- `Health check`: la URL de la demo seguida de `/health`.
- `Panel de MLflow`: interfaz para consultar las métricas registradas.

Las direcciones terminan en `trycloudflare.com`, cambian con cada sesión y dejan de funcionar cuando se cierra Colab.

### 6. Comprobar el despliegue desde afuera

Ejecutar la celda de la sección **Comprobar desde afuera**. La comprobación pasa por la URL pública y realiza cuatro verificaciones:

1. Consulta `GET /health` y comprueba que responda con HTTP 200.
2. Envía una consulta de prueba a la API de Gradio `/responder`.
3. Comprueba que el panel público de MLflow responda.
4. Muestra la última línea de `peticiones.jsonl`.

Después de ejecutar la celda, abrir la URL de la demo desde otro dispositivo, preferiblemente un teléfono con datos móviles. También conviene abrir `/health` y el panel de MLflow desde ese dispositivo.

### 7. Mantener el servicio disponible

Ejecutar la celda de la sección **Mantenerla en línea** y dejarla activa durante la demostración. Cada minuto consulta `/health` e imprime el estado, el número de peticiones, las activaciones de la válvula, los errores y el estado de MLflow.

Colab puede desconectarse por inactividad o por límite de GPU. Por eso la sesión debe abrirse y comprobarse unos minutos antes de la sustentación, con un video de respaldo disponible.

### 8. Apagar los procesos

Al terminar, interrumpir la celda de monitoreo y ejecutar la última celda. Esta detiene la aplicación, MLflow y ambos túneles de Cloudflare.

## Health check

El health check público es `GET /health`. No genera una respuesta del modelo ni consume una consulta real. Informa:

- `status`: `cargando`, `ok`, `degradado` o `error`.
- Versión, dispositivo y decoder usados.
- Tiempo activo del proceso.
- Estado de MLflow.
- Número de peticiones, activaciones de la válvula y errores.

Cuando la carga termina, también incorpora la comprobación interna de los modelos y del índice realizada por `final/sistema.py`.

## Logging y observabilidad

Cada petición atendida deja una línea JSON en `peticiones.jsonl`. El registro contiene la hora, un identificador corto, la versión, el dispositivo, el decoder, los componentes utilizados, la decisión del agente, la válvula, el número de referencias, la longitud de la consulta, las latencias y el tipo de error si existe.

El texto de la consulta y de la respuesta no se guarda. Solo se registra la longitud de la consulta. Esto evita enviar a los registros o a MLflow datos personales escritos por el usuario.

Cuando `MLFLOW_TRACKING_URI` está configurado, la aplicación registra por petición:

- Latencia total y latencias por componente.
- Activación de la válvula.
- Presencia de error.
- Número acumulado de peticiones.
- Versión, dispositivo y decoder como parámetros del run.

MLflow usa una base SQLite local a la sesión. Tanto la base como los logs desaparecen cuando se elimina el entorno de Colab, salvo que se descarguen manualmente.

## Limitaciones operativas

- No hay una URL permanente: las direcciones cambian al reiniciar Colab.
- El servicio solo está disponible mientras la sesión y los túneles siguen activos.
- El panel público de MLflow debe compartirse únicamente durante la revisión o sustentación.
- La versión desplegada usa el sistema M3 sin cuantizar en GPU, porque las variantes cuantizadas no cumplieron el umbral de calidad y latencia definido para la técnica de eficiencia.
- El sistema final es textual. El componente visual del Módulo 4 no se implementó y no es una exclusión causada por el despliegue.
