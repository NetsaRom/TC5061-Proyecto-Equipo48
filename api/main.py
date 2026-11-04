
import os
import requests

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ==========================================
# CONFIGURACIÓN DE MLFLOW MODEL SERVING
# ==========================================

MODEL_NAME = "mibici-equipo48-p2-riesgo"
MODEL_ALIAS = "champion"

MLFLOW_SERVING_URL = os.getenv(
    "MLFLOW_SERVING_URL",
    "http://host.containers.internal:5001"
)

FEATURES = [
    "promedio_duracion_salida",
    "pct_viajes_matutinos",
    "viajes_entrantes"
]


# ==========================================
# ESQUEMA DE ENTRADA - PYDANTIC
# ==========================================

class PredictionInput(BaseModel):

    promedio_duracion_salida: float = Field(
        ge=0,
        description="Duración promedio de los viajes de salida"
    )

    pct_viajes_matutinos: float = Field(
        ge=0,
        le=1,
        description="Proporción de viajes matutinos"
    )

    viajes_entrantes: float = Field(
        ge=0,
        description="Cantidad de viajes entrantes"
    )


# ==========================================
# ESQUEMA DE SALIDA
# ==========================================

class PredictionOutput(BaseModel):

    modelo: str
    prediccion: int
    descripcion: str


# ==========================================
# CREACIÓN DE LA API
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando API MiBici...")
    print(f"MLflow Model Serving: {MLFLOW_SERVING_URL}")
    yield
    print("Finalizando API MiBici...")


app = FastAPI(
    title="MiBici - API de Predicción",
    description=(
        "API REST para predecir alta demanda "
        "en estaciones de MiBici utilizando el modelo P2."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# ENDPOINT DE SALUD
# ==========================================

@app.get("/health")
def health():

    try:
        response = requests.get(
            f"{MLFLOW_SERVING_URL}/ping",
            timeout=5
        )

        response.raise_for_status()

        return {
            "status": "ok",
            "modelo": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "serving": "disponible"
        }

    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="MLflow Model Serving no está disponible."
        )


# ==========================================
# ENDPOINT DE PREDICCIÓN
# ==========================================

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):

    try:

        # Preparar datos para MLflow
        payload = {
            "dataframe_split": {
                "columns": FEATURES,
                "data": [[
                    data.promedio_duracion_salida,
                    data.pct_viajes_matutinos,
                    data.viajes_entrantes
                ]]
            }
        }

        # Enviar solicitud al modelo
        response = requests.post(
            f"{MLFLOW_SERVING_URL}/invocations",
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        # Obtener predicción
        result = response.json()
        prediction = int(result["predictions"][0])

        descripcion = (
            "Alta demanda"
            if prediction == 1
            else "Demanda no alta"
        )

        return PredictionOutput(
            modelo=MODEL_NAME,
            prediccion=prediction,
            descripcion=descripcion
        )

    except requests.RequestException as error:
        raise HTTPException(
            status_code=503,
            detail=f"Error al comunicarse con MLflow: {str(error)}"
        )

    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"Respuesta inesperada de MLflow: {str(error)}"
        )
