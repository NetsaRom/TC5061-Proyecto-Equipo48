# Despliegue del modelo P2 mediante FastAPI y Docker

## 1. Descripción general

Como parte del proyecto MiBici, se implementó un servicio REST para realizar predicciones de alta demanda en estaciones del sistema de bicicletas públicas.

La solución utiliza el modelo de clasificación P2, registrado en MLflow Model Registry, y permite consultar sus predicciones mediante una API desarrollada con FastAPI y desplegada en un contenedor OCI compatible mediante Docker.

La arquitectura está integrada por dos servicios independientes:

1. **MLflow Model Serving:** publica el modelo registrado y recibe solicitudes de inferencia.
2. **FastAPI:** expone endpoints REST, valida los datos de entrada y envía las solicitudes al servicio de MLflow.

FastAPI funciona como intermediario entre el cliente y el modelo de aprendizaje automático.


## 2. Arquitectura de la solución

```text
                        Cliente
                           |
                           | HTTP/JSON
                           v
              +--------------------------+
              |        FastAPI           |
              |   Contenedor Docker      |
              |      Puerto 8000         |
              |                          |
              |  GET  /health            |
              |  POST /predict           |
              +------------+-------------+
                           |
                           | dataframe_split
                           | HTTP/JSON
                           v
              +--------------------------+
              | MLflow Model Serving     |
              |      Puerto 5001         |
              |                          |
              |  GET  /ping              |
              |  POST /invocations       |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              | Modelo registrado P2     |
              | mibici-equipo48-p2-riesgo|
              | Alias: champion          |
              +--------------------------+
```

### Flujo de una predicción

1. El cliente envía los datos de una estación a `POST /predict`.
2. FastAPI valida la estructura y los rangos mediante Pydantic.
3. La API convierte la solicitud al formato `dataframe_split`.
4. FastAPI envía la solicitud a `POST /invocations` de MLflow Model Serving.
5. MLflow carga y ejecuta el modelo registrado con el alias `champion`.
6. MLflow devuelve una lista de predicciones.
7. FastAPI valida la respuesta y entrega un JSON estable al cliente.

## 3. Modelo utilizado
El servicio utiliza el modelo de clasificación P2 registrado en MLflow Model Registry.

| Propiedad | Valor |
|---|---|
| Nombre del modelo | `mibici-equipo48-p2-riesgo` |
| Alias | `champion` |
| Tipo de problema | Clasificación |
| Variable objetivo | `alta_demanda` |
| Clase 0 | Demanda no alta |
| Clase 1 | Alta demanda |

El modelo recibe las siguientes características:

| Variable | Descripción |
|---|---|
| `promedio_duracion_salida` | Duración promedio de los viajes de salida. |
| `pct_viajes_matutinos` | Proporción de viajes realizados durante la mañana. |
| `viajes_entrantes` | Cantidad de viajes entrantes a la estación. |


## 4. Estructura principal del repositorio

La implementacion utiliza la siguiente escritura:
TC5061-Proyecto-Equipo48/
│
├── api/
│   ├── main.py
│   └── requirements.txt
│
├── Containerfile
│
├── requirements.txt
│
└── README.md

### Descripción de los archivos

- **api/main.py:** contiene la implementación de FastAPI, los esquemas de validación y los endpoints `/health` y `/predict`.
- **api/requirements.txt:** contiene las dependencias necesarias para ejecutar la API.
- **Containerfile:** define la construcción de la imagen del contenedor.
- **requirements.txt:** conserva las dependencias generales del proyecto.

## 5. Requisitos previos

Antes de ejecutar el proyecto se requiere:

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic
- Requests
- MLflow Model Serving
- Docker/OCI compatible Containerfile
- Docker Desktop para Mac con procesador Intel, instalado y en ejecución.
- Dependencias instaladas desde `requirements.txt`.
- Datos procesados para ejecutar los entrenamientos.
- Modelo P2 registrado en MLflow Model Registry con el alias `champion`.

Este documento está ajustado para macOS Monterey 12.7.6 en una Mac con procesador Intel Core i7. Los comandos se ejecutan desde Terminal con zsh o bash. No se requiere WSL, PowerShell ni Rosetta 2 para esta plataforma Intel.

## 5.1 Compatibilidad de Docker Desktop con macOS Monterey

El equipo objetivo es una Mac Intel Core i7 con macOS Monterey 12.7.6. Docker Desktop ofrece instaladores separados para Mac con procesador Intel. Sin embargo, las versiones actuales de Docker Desktop solo reciben soporte oficial en la versión actual de macOS y las dos versiones mayores anteriores. Por ello, Monterey puede requerir una versión histórica de Docker Desktop compatible o una actualización de macOS.

Antes de continuar, verificar la instalación:

```bash
uname -m
sw_vers
docker --version
docker compose version
docker info
```

Resultados esperados:

- `uname -m` debe mostrar `x86_64`.
- `sw_vers` debe mostrar macOS 12.7.6.
- `docker info` debe finalizar sin error y mostrar información del Docker Engine.
- Se recomienda asignar al menos 4 GB de memoria a Docker Desktop. Para MLflow, FastAPI y pruebas simultáneas es preferible usar entre 6 y 8 GB si la Mac dispone de memoria suficiente.

Si Docker Desktop no inicia en Monterey, no se debe instalar ciegamente la versión más reciente. Debe utilizarse una versión histórica documentada como compatible con macOS 12 o actualizar el sistema operativo antes de ejecutar la integración.

## 6. Configuración del entorno

Para ejecutar la solución se requiere:

- Python 3.11.
- MLflow instalado en el entorno donde se ejecutará Model Serving.
- Podman instalado y funcionando.
- Modelo P2 registrado en MLflow Model Registry.
- Acceso a los artefactos del modelo registrados en MLflow.

## 7. Ejecución de MLflow Model Serving

Inicia nuevo servidor en otra terminal dentro del mismo ambiente:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow-tracking.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

## Entrena y registra el modelo P2

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
mlflow run . -e experiment_p2 --experiment-name mibici-equipo48 --env-manager local
python register_p2_model.py
```


### Características de entrada

| Variable | Tipo | Restricción | Descripción |
|---|---:|---:|---|
| `promedio_duracion_salida` | Número | `>= 0` | Duración promedio de los viajes de salida. |
| `pct_viajes_matutinos` | Número | Entre `0` y `1` | Proporción de viajes realizados durante la mañana. |
| `viajes_entrantes` | Número | `>= 0` | Cantidad de viajes entrantes a la estación. |

### Variable objetivo

- `0`: demanda no alta.
- `1`: alta demanda.

La etiqueta `alta_demanda` es un proxy construido por el proyecto a partir de datos históricos. No debe interpretarse automáticamente como telemetría física en tiempo real ni como garantía de demanda futura.



## 8. Contrato de comunicación FastAPI - MLflow

### Transporte

- Protocolo: HTTP.
- Formato: JSON UTF-8.
- URL base configurable con `MLFLOW_SERVING_URL`.
- Timeout configurable con `MLFLOW_TIMEOUT_SECONDS`.
- Disponibilidad de MLflow: `GET /ping`.
- Inferencia de MLflow: `POST /invocations`.

### Solicitud enviada por FastAPI a MLflow

```json
{
  "dataframe_split": {
    "columns": [
      "promedio_duracion_salida",
      "pct_viajes_matutinos",
      "viajes_entrantes"
    ],
    "data": [
      [9.2547, 0.2379, 1452]
    ]
  }
}
```

### Respuesta esperada de MLflow

```json
{
  "predictions": [0]
}
```

La API verifica que:

- Exista la propiedad `predictions`.
- Sea una lista.
- Contenga exactamente una predicción.
- La predicción sea convertible a entero.
- El valor final sea `0` o `1`.

## 9. Iniciar MLflow Model Serving

Con el servidor de tracking activo y `MLFLOW_TRACKING_URI` configurado:

```bash
mlflow models serve \
    -m "models:/mibici-equipo48-p2-riesgo@champion" \
    --host 0.0.0.0 \
    --port 5001 \
    --env-manager local
```

Mantener esta terminal abierta y corriendo.

### Verificar MLflow Model Serving

```bash
curl -i http://127.0.0.1:5001/ping
```

La respuesta debe incluir un estado HTTP exitoso. Si este paso falla, FastAPI tampoco podrá ejecutar predicciones.

## 10. Construcción de la API con Docker

Desde la raíz del repositorio:

```bash
docker build -t mibici-api-p2 -f Containerfile .
```

Verificar la imagen:

```bash
docker images
```


## 11. Ejecución del contenedor

```bash
docker run -d \
  --name mibici-api-p2 \
  -p 8000:8000 \
  -e MLFLOW_SERVING_URL=http://host.docker.internal:5001 \
  -e MLFLOW_TIMEOUT_SECONDS=10 \
  mibici-api-p2
```

`host.docker.internal` permite que FastAPI, ejecutándose dentro del contenedor Docker Desktop, acceda a MLflow Model Serving en macOS. Dentro del contenedor, `localhost` apuntaría al propio contenedor y no a la Mac.

### Verificar el contenedor

```bash
docker ps
docker logs mibici-api-p2
```

Si el nombre ya está ocupado:

```bash
docker rm -f mibici-api-p2
```

Después se puede ejecutar nuevamente el comando `docker run`.

## 12. Endpoints de FastAPI

### 12.1 `GET /health`

Comprueba que FastAPI pueda comunicarse con MLflow Model Serving.

```bash
curl -i http://127.0.0.1:8000/health
```

Respuesta esperada:
al colocarle la -i nos indicara el estatus del codigo de estado
```bash
HTTP/1.1 200 OK
date: Sun, 08 Nov 2026 02:44:30 GMT
server: uvicorn
content-length: 89
content-type: application/json
```

```json
{
  "status": "ok",
  "modelo": "mibici-equipo48-p2-riesgo",
  "alias": "champion",
  "serving": "disponible"
}
```

### 12.2 `POST /predict`

Ejecuta una predicción de alta demanda.

```bash
curl i- -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "promedio_duracion_salida":9.2547,
    "pct_viajes_matutinos":0.2379,
    "viajes_entrantes":1452
    }'
```

Respuesta válida de ejemplo:

```json
{
  "modelo": "mibici-equipo48-p2-riesgo",
  "alias": "champion",
  "prediccion": 0,
  "descripcion": "Demanda no alta"
}
```

El resultado puede ser `0` o `1`, según los datos y la versión del modelo asociada al alias `champion`.

### Documentación interactiva

Con el contenedor activo:

```text
http://127.0.0.1:8000/docs
```

Swagger UI permite consultar los esquemas y ejecutar solicitudes de prueba.

## 13. Códigos de respuesta

| Código | Significado |
|---:|---|
| `200` | Solicitud procesada correctamente. |
| `422` | Los datos no cumplen el esquema o los rangos permitidos. |
| `502` | MLflow respondió con error o incumplió el contrato esperado. |
| `503` | MLflow Model Serving no está disponible. |
| `504` | Se agotó el tiempo de espera al consultar MLflow. |


### 14 Pruebas de extremo a extremo

Antes de ejecutarlas se requiere:

1. MLflow Tracking activo en el puerto `5000`, cuando el registro depende de ese servidor.
2. MLflow Model Serving activo en el puerto `5001`.
3. Contenedor FastAPI activo en el puerto `8000`.

macOS Monterey:

```bash
export API_BASE_URL="http://127.0.0.1:8000"
python -m pytest tests/test_e2e.py -v
python scripts/validate_e2e.py
```

Las pruebas E2E validan:

- Respuesta HTTP 200 de `/health`.
- Predicción HTTP 200 desde `/predict`.
- Presencia del nombre y alias del modelo.
- Predicción limitada a `0` o `1`.
- Respuesta HTTP 422 para datos inválidos.

El script `scripts/validate_e2e.py` genera:

```text
reports/evidencias/e2e_resultados.json
```

La integración solamente se considera aprobada cuando el archivo contiene:

```json
{
  "resultado": "APROBADO"
}
```

## 15. Solución de problemas

### `/health` devuelve 503

Posibles causas:

- MLflow Model Serving no está activo.
- El puerto `5001` no está publicado o está ocupado.
- Se utilizó `localhost` dentro del contenedor.
- El firewall bloquea la comunicación.

Verificaciones:

```bash
curl -i http://127.0.0.1:5001/ping
docker logs mibici-api-p2
```

Confirma que el contenedor tenga:

```text
MLFLOW_SERVING_URL=http://host.docker.internal:5001
```

### `/predict` devuelve 502

Revisar:

- Nombres y orden de las columnas.
- Firma del modelo registrado.
- Estructura `dataframe_split`.
- Formato de `predictions`.
- Registros de MLflow Model Serving.

### `/predict` devuelve 422

Uno o más campos faltan o están fuera del rango permitido. Ejemplo válido:

```json
{
  "promedio_duracion_salida": 9.2547,
  "pct_viajes_matutinos": 0.2379,
  "viajes_entrantes": 1452
}
```

### El contenedor no inicia

```bash
docker ps -a
docker logs mibici-api-p2
```

Si existe un contenedor anterior con el mismo nombre:

```bash
docker rm -f mibici-api-p2
```

### El modelo no puede cargarse

Confirmar:

- Que `MLFLOW_TRACKING_URI` apunta al registro correcto.
- Que existe el modelo `mibici-equipo48-p2-riesgo`.
- Que una versión tiene asignado el alias `champion`.
- Que los artefactos de esa versión son accesibles.
- Que el entorno contiene las dependencias usadas durante el entrenamiento.

## 16. Detener y limpiar el despliegue

Detener el contenedor:

```bash
docker stop mibici-api-p2
```

Eliminar el contenedor:

```bash
docker rm mibici-api-p2
```

Eliminar la imagen:

```bash
docker rmi mibici-api-p2
```

Los servidores de MLflow se detienen con `Ctrl + C` en las terminales donde fueron iniciados.

## 17. Seguridad y reproducibilidad

- No almacenar contraseñas, tokens ni credenciales en el repositorio.
- Utilizar variables de entorno para las URLs y configuraciones sensibles.
- No versionar bases de tracking, artefactos o entornos virtuales.
- Mantener `requirements.txt` y los archivos de entorno sincronizados.
- Registrar parámetros, métricas, firma del modelo y ejemplo de entrada.
- Conservar el hash del manifiesto de datos y el commit de Git utilizado.
- Revisar dependencias con `pip-audit`.
- Analizar código Python con `bandit`.
- Ejecutar `python -m pip check` antes de integrar cambios.

Ejemplo:

```bash
python -m pip install pip-audit bandit
pip-audit -r requirements.txt
bandit -r mibici_mlops api
```

