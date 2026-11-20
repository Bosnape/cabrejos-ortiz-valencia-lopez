# Despliegue en Google Colab

El sistema final corre en una GPU T4 de Colab y se publica con dos túneles temporales de Cloudflare: uno para la app y otro
para el panel de MLflow. Se eligió Colab porque en CPU una consulta tarda cerca de dos minutos y la GPU gratuita de Hugging
Face Spaces no alcanzó para mantener el despliegue. No hacen falta cuentas, tokens ni claves.

## Pasos

1. Abrir [`despliegue_en_colab.ipynb`](despliegue_en_colab.ipynb) en Colab y elegir **GPU T4** (Entorno de ejecución,
   Cambiar tipo de entorno). El notebook clona `main`, así que los cambios deben estar integrados ahí.
2. Ejecutar las celdas en orden. Primero se instala el entorno; después arrancan el panel de MLflow (puerto 5000) y la app
   (puerto 7860). La app carga los modelos en segundo plano y la celda espera hasta que `/health` responda `ok` (hasta 6 min).
3. La celda de túneles imprime tres direcciones `trycloudflare.com`: la demo, su `/health` y el panel de MLflow.
4. La celda de comprobación consulta `/health` y hace una pregunta de prueba por la ruta pública, y confirma que el panel
   responde. Conviene abrir también la demo desde un teléfono con datos móviles.
5. Dejar corriendo la celda de monitoreo durante la demo: imprime cada minuto el estado, las peticiones, la válvula y los errores.
6. Al terminar, interrumpir esa celda y ejecutar la última, que detiene la app, MLflow y los túneles.

Si algo falla, los logs quedan en `mlflow.log`, `app.log` y `cloudflared_*.log`.

## Health check

`GET /health` no genera texto: revisa que los modelos hayan cargado y que el índice responda. Devuelve `status`
(`cargando`, `ok`, `degradado` o `error`), versión, dispositivo, decoder, tiempo activo, estado de MLflow y los contadores de
peticiones, válvula y errores.

## Log y observabilidad

- Cada petición deja una línea JSON en `peticiones.jsonl`: hora, id, versión, dispositivo, decoder, componentes usados,
  acción del agente, válvula, número de referencias, longitud de la consulta, latencias y error. No se guarda el texto de la
  consulta ni de la respuesta.
- Con `MLFLOW_TRACKING_URI` definida (el notebook la define), la app registra por petición la latencia total y por componente,
  la válvula, el error y el total de peticiones. Versión, dispositivo y decoder quedan como parámetros del run.
- Para ver las métricas: abrir el panel, entrar al experimento `asistente-legal-laboral` y al run `app-v<versión>-cuda`.

## Limitaciones

- Las URL cambian con cada sesión y dejan de funcionar al cerrarla; Colab puede desconectar por inactividad o límite de GPU.
  Abrir la sesión unos minutos antes de la sustentación y tener el video de respaldo.
- La base de MLflow y los logs viven en la sesión: descargar `mlflow.db` y `peticiones.jsonl` antes de cerrarla.
- Se despliega M3 sin cuantizar porque las variantes cuantizadas no cumplieron el umbral de la técnica de eficiencia.
- El sistema es solo textual; el componente visual del Módulo 4 no se implementó (ver el reporte, sección 8).
