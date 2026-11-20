from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field
import requests

# ==========================================
# CONFIGURACIÓN DEL MODELO
# ==========================================

MODEL_NAME = "mibici-equipo48-p2-riesgo"
MODEL_VERSION = "1"

FEATURES = ["promedio_duracion_salida", "pct_viajes_matutinos", "viajes_entrantes"]

# Artefacto P2 empaquetado en el repositorio (modo local, Semana 8).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "p2"

MODE_REMOTE = "remoto"
MODE_LOCAL = "local"


def get_mlflow_serving_url() -> str:
    """Obtiene la URL de MLflow Model Serving; vacía significa modo local."""
    return os.getenv("MLFLOW_SERVING_URL", "").strip().rstrip("/")


def get_serving_mode() -> str:
    """Modo remoto si hay URL de MLflow Model Serving; si no, modo local."""
    return MODE_REMOTE if get_mlflow_serving_url() else MODE_LOCAL


def get_mlflow_timeout_seconds() -> float:
    """Obtiene el tiempo máximo de espera para la comunicación con MLflow."""
    return float(os.getenv("MLFLOW_TIMEOUT_SECONDS", "10"))


def check_model_serving() -> None:
    """Comprueba que MLflow Model Serving responda antes de atender tráfico."""
    response = requests.get(
        f"{get_mlflow_serving_url()}/ping",
        timeout=get_mlflow_timeout_seconds(),
    )
    response.raise_for_status()


def load_local_model():
    """Carga el artefacto P2 con MLflow pyfunc dentro del proceso de la API."""
    import mlflow.pyfunc

    return mlflow.pyfunc.load_model(str(MODEL_PATH))


def predict_remote(dataframe: pd.DataFrame) -> list:
    """Envía la predicción a MLflow Model Serving por HTTP."""
    response = requests.post(
        f"{get_mlflow_serving_url()}/invocations",
        json={
            "dataframe_split": {
                "columns": FEATURES,
                "data": dataframe.values.tolist(),
            }
        },
        timeout=get_mlflow_timeout_seconds(),
    )
    response.raise_for_status()
    return response.json().get("predictions")


def predict_local(dataframe: pd.DataFrame) -> list:
    """Ejecuta la predicción con el modelo cargado en memoria."""
    return list(app.state.model.predict(dataframe))


# ==========================================
# ESQUEMA DE ENTRADA - PYDANTIC
# ==========================================


class PredictionInput(BaseModel):
    promedio_duracion_salida: float = Field(
        ge=0, description="Duración promedio de los viajes de salida"
    )

    pct_viajes_matutinos: float = Field(ge=0, le=1, description="Proporción de viajes matutinos")

    viajes_entrantes: float = Field(ge=0, description="Cantidad de viajes entrantes")


# ==========================================
# ESQUEMA DE SALIDA
# ==========================================


class PredictionOutput(BaseModel):
    modelo: str
    version: str
    prediccion: int
    descripcion: str


# ==========================================
# CARGA DEL MODELO
# ==========================================


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Iniciando API MiBici...")
    app.state.serving_mode = get_serving_mode()

    if app.state.serving_mode == MODE_REMOTE:
        # Docker Compose: el modelo vive en el servicio model-serving.
        print(f"Conectando con MLflow Model Serving: {get_mlflow_serving_url()}")
        try:
            check_model_serving()
            print("MLflow Model Serving está disponible.")
        except requests.RequestException as error:
            print(f"Error al conectar con MLflow Model Serving: {error}")
            raise
    else:
        # Contenedor único (docker run): el modelo se carga en la API.
        print(f"MLFLOW_SERVING_URL no definida. Cargando modelo local: {MODEL_PATH}")
        app.state.model = load_local_model()
        print("Modelo P2 cargado correctamente.")

    yield

    print("Finalizando API MiBici...")


# ==========================================
# CREACIÓN DE LA API
# ==========================================

app = FastAPI(
    title="MiBici - API de Predicción",
    description=(
        "API REST para predecir alta demanda en estaciones de MiBici utilizando el modelo P2."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ==========================================
# ENDPOINT DE SALUD
# ==========================================


@app.get("/health")
def health():

    mode = getattr(app.state, "serving_mode", get_serving_mode())

    if mode == MODE_REMOTE:
        try:
            check_model_serving()
        except requests.RequestException as error:
            raise HTTPException(
                status_code=503, detail="MLflow Model Serving no está disponible."
            ) from error
    elif getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado.")

    return {
        "status": "ok",
        "modelo": MODEL_NAME,
        "version": MODEL_VERSION,
        "serving": "disponible",
        "modo": mode,
    }


# ==========================================
# ENDPOINT DE PREDICCIÓN
# ==========================================


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):

    mode = getattr(app.state, "serving_mode", get_serving_mode())

    try:
        dataframe = pd.DataFrame(
            [[data.promedio_duracion_salida, data.pct_viajes_matutinos, data.viajes_entrantes]],
            columns=FEATURES,
            dtype="float32",
        )

        predictions = predict_remote(dataframe) if mode == MODE_REMOTE else predict_local(dataframe)

        if not isinstance(predictions, list) or len(predictions) != 1:
            raise ValueError("El modelo devolvió una respuesta inválida.")

        prediction = int(predictions[0])

        descripcion = "Alta demanda" if prediction == 1 else "Demanda no alta"

        return PredictionOutput(
            modelo=MODEL_NAME,
            version=MODEL_VERSION,
            prediccion=prediction,
            descripcion=descripcion,
        )

    except requests.RequestException as error:
        raise HTTPException(
            status_code=503, detail="MLflow Model Serving no está disponible."
        ) from error

    except (TypeError, ValueError, KeyError) as error:
        raise HTTPException(
            status_code=502, detail=f"Error al ejecutar la predicción: {error!s}"
        ) from error
