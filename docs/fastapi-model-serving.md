# FastAPI Model Serving — MiBici MLOps

## 1. Objetivo

Este documento describe la implementación del servicio de *Model Serving* para el proyecto MiBici MLOps. El servicio expone mediante una API REST el modelo de regresión de demanda registrado en MLflow Model Registry.

La API está desarrollada con **FastAPI** y permite:

- Cargar una versión registrada del modelo desde MLflow Model Registry.
- Consultar el estado de disponibilidad del servicio y del modelo.
- Recibir datos de entrada para obtener predicciones.
- Validar los datos recibidos mediante esquemas Pydantic.
- Consultar la documentación interactiva de los endpoints.

Esta implementación corresponde a un servicio de ejecución local para el proyecto. No representa, por sí sola, un despliegue productivo.

## 2. Arquitectura

La implementación está organizada en el paquete `mibici_mlops/api/`.

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Define la aplicación FastAPI, el ciclo de vida de carga del modelo y los endpoints HTTP. |
| `schemas.py` | Define los esquemas de entrada y salida, así como las reglas de validación de datos. |
| `model.py` | Configura MLflow, carga el modelo desde Model Registry y ejecuta las predicciones. |
| `__init__.py` | Identifica el directorio como un paquete Python. |

### Flujo general de una predicción

1. El cliente envía una solicitud HTTP `POST` al endpoint `/predict`.
2. FastAPI valida el cuerpo de la solicitud con el esquema `PredictionRequest`.
3. La capa de serving comprueba que el modelo esté disponible.
4. El modelo registrado en MLflow ejecuta la predicción con las características recibidas.
5. La API devuelve la predicción y los metadatos del modelo en un objeto `PredictionResponse`.

## 3. Modelo registrado en MLflow

La implementación utiliza el modelo de regresión de demanda registrado en MLflow Model Registry.

| Propiedad | Valor |
|---|---|
| Nombre del modelo | `mibici-equipo48-p1-demanda` |
| Versión configurada por defecto | `1` |
| URI configurada por defecto | `models:/mibici-equipo48-p1-demanda/1` |
| Tipo de problema | Regresión |
| Algoritmo documentado en el proyecto | `RandomForestRegressor` |
| Experimento relacionado | `mibici-equipo48-p1-fastapi` |

La versión configurada por defecto es la versión `1`. La URI puede modificarse mediante la variable de entorno `MLFLOW_MODEL_URI`, siempre que la URI alternativa sea válida y accesible desde el entorno donde se ejecuta la API.

> **Importante:** el README principal del proyecto también documenta el alias `champion` para el modelo registrado. Sin embargo, el valor predeterminado de esta implementación de FastAPI es la URI versionada `models:/mibici-equipo48-p1-demanda/1`, no el alias `champion`.

## 4. Configuración del entorno

### 4.1. Dependencias

La implementación utiliza FastAPI y Uvicorn, además de las dependencias de MLflow y del proyecto.

Las dependencias añadidas para la API son:

```text
fastapi==0.162.0
uvicorn==0.54.0
```

Se recomienda ejecutar los comandos desde la raíz del repositorio y con el entorno virtual del proyecto activado.

### 4.2. Variables de entorno

| Variable | Descripción |
|---|---|
| `MLFLOW_TRACKING_URI` | URI del servidor o almacén de seguimiento de MLflow que contiene el experimento y el Model Registry. |
| `MLFLOW_MODEL_URI` | URI del modelo que debe cargarse desde MLflow Model Registry. |

Si no se especifica `MLFLOW_MODEL_URI`, la aplicación utiliza:

```text
models:/mibici-equipo48-p1-demanda/1
```

Si no se especifica `MLFLOW_TRACKING_URI`, el código utiliza una base SQLite local llamada `mlflow.db`, ubicada en la raíz del proyecto.

**Nota sobre el tracking de MLflow:** la API debe apuntar explícitamente al mismo tracking store que contiene el modelo y sus artefactos. No debe asumirse que dos archivos SQLite distintos contienen el mismo registro.

### 4.3. Ejecución local

Desde la raíz del repositorio, con el entorno virtual activado:

```bash
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
export MLFLOW_MODEL_URI="models:/mibici-equipo48-p1-demanda/1"

uvicorn mibici_mlops.api.main:app --host 127.0.0.1 --port 8000
```

Ajusta `MLFLOW_TRACKING_URI` si el modelo se encuentra en otro tracking store o servidor de MLflow.

La aplicación queda disponible localmente en:

```text
http://127.0.0.1:8000
```

## 5. Endpoints

### 5.1. `GET /health`

Permite consultar el estado de la API y comprobar si el modelo se encuentra cargado.

Ejemplo:

```bash
curl http://127.0.0.1:8000/health
```

La respuesta informa el estado del servicio y la disponibilidad del modelo. Si el modelo no está disponible, el endpoint comunica esa condición mediante el código HTTP correspondiente definido por la implementación.

### 5.2. `POST /predict`

Recibe las características de una observación y devuelve la predicción de demanda del modelo.

Ejemplo de solicitud:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "estacion": "TEST",
    "hora_dia": 10,
    "dia_semana": 2,
    "mes": 9,
    "fin_semana": 0,
    "viajes_lag_1": 3.0,
    "viajes_lag_24": 4.0,
    "viajes_lag_168": 5.0,
    "promedio_24h": 4.0
  }'
```

Ejemplo de respuesta:

```json
{
  "prediction": 4.475693865204796,
  "unit": "viajes",
  "model_name": "mibici-equipo48-p1-demanda",
  "model_version": "1"
}
```

El valor numérico mostrado corresponde a una ejecución de prueba y puede variar según el modelo cargado y los datos de entrada.

### 5.3. `GET /docs`

FastAPI genera documentación interactiva basada en OpenAPI.

Con el servidor en ejecución, visita:

```text
http://127.0.0.1:8000/docs
```

Desde esta interfaz se pueden consultar los esquemas y probar los endpoints.

## 6. Esquemas y validación

Los esquemas de solicitud y respuesta se encuentran en `mibici_mlops/api/schemas.py`.

- `PredictionRequest`: define las características que debe proporcionar el cliente para solicitar una predicción.
- `PredictionResponse`: define la estructura de la respuesta, incluyendo la predicción y los metadatos del modelo.
- `HealthResponse`: define la estructura de la respuesta del endpoint de salud.

La validación se realiza mediante Pydantic. Los esquemas restringen los campos admitidos y aplican las reglas de validación definidas para las características de entrada. Los campos adicionales no contemplados por el esquema se rechazan.

La solicitud debe respetar los nombres, tipos y restricciones definidos en el código. El ejemplo de payload de este documento sirve como referencia y no sustituye los esquemas implementados.

## 7. Manejo de errores

La API contempla, entre otros, los siguientes escenarios:

| Código HTTP | Significado |
|---|---|
| `200` | La solicitud se procesó correctamente. |
| `422` | Los datos enviados no cumplen el esquema o las reglas de validación. |
| `503` | El modelo no está disponible para atender la predicción. |
| `500` | Ocurrió un error durante la ejecución de la predicción. |

La carga del modelo se realiza durante el ciclo de vida de la aplicación. Si la carga no puede completarse, el proceso de la API puede permanecer activo, pero las solicitudes de predicción no podrán procesarse hasta que el modelo esté disponible.

## 8. Pruebas y calidad

La implementación incluye pruebas unitarias para la API y la capa de carga y ejecución del modelo:

```text
tests/test_api.py
tests/test_model.py
```

Comandos utilizados para validar la implementación:

```bash
pytest -v
ruff check mibici_mlops/api tests/test_api.py tests/test_model.py
ruff format --check mibici_mlops/api tests/test_api.py tests/test_model.py
```

Resultados registrados durante la validación de esta implementación:

- **31 pruebas aprobadas**.
- **Cobertura del 100 %** en los módulos de `mibici_mlops/api` durante la medición realizada.
- Ruff: todas las verificaciones de linting aprobadas.
- Ruff format: los archivos revisados ya estaban formateados.

Durante la ejecución de las pruebas se observó una advertencia de deprecación relacionada con el uso de `httpx` en `starlette.testclient`. La advertencia no impidió que las pruebas finalizaran correctamente. Conviene revisarla al actualizar las dependencias de testing, sin cambiar versiones de forma aislada ni asumir compatibilidad sin volver a ejecutar la suite.

## 9. Limitaciones y consideraciones operativas

- La implementación está preparada para ejecución local y pruebas del proyecto; no incluye un despliegue productivo.
- El entorno que ejecuta FastAPI debe poder acceder al tracking store de MLflow y a los artefactos del modelo.
- Si MLflow utiliza almacenamiento remoto, también deben estar configuradas las credenciales y variables necesarias para acceder a dicho almacenamiento.
- La URI configurada debe apuntar a un modelo y una versión disponibles en el Registry.
- La disponibilidad del endpoint no garantiza por sí misma que el modelo esté cargado; debe verificarse la respuesta de `/health`.
- La API utiliza la versión `1` del modelo por defecto. Para cambiar el modelo o la versión, configura `MLFLOW_MODEL_URI` y valida el resultado con pruebas.
- El servicio no incorpora autenticación, autorización, escalamiento horizontal ni políticas de observabilidad productiva en esta entrega.

## 10. Estructura de archivos relacionada

```text
mibici_mlops/
└── api/
    ├── __init__.py
    ├── main.py
    ├── model.py
    └── schemas.py

tests/
├── test_api.py
└── test_model.py

docs/
└── fastapi-model-serving.md
```

## 11. Referencias dentro del repositorio

- [README principal](../README.md)
- [Índice de documentación de modelos](README.md)
- [Documentación del modelo de regresión](modelo_regresion.md)
- [Documentación del modelo de clasificación](modelo_clasificacion.md)
- [Documentación del modelo de clustering](modelo_clustering.md)

---

**Proyecto:** MiBici MLOps — Equipo 48
**Componente:** FastAPI Model Serving
**Rama de trabajo:** `feat/fastapi-model-serving`
