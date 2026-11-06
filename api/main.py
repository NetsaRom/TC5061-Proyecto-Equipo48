
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ==========================================
# CONFIGURACIÓN DEL MODELO
# ==========================================

MODEL_NAME = "mibici-equipo48-p2-riesgo"
MODEL_VERSION = "1"

# Ruta del modelo empaquetado dentro del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "p2"

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
    version: str
    prediccion: int
    descripcion: str


# ==========================================
# CARGA DEL MODELO
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Iniciando API MiBici...")
    print(f"Cargando modelo desde: {MODEL_PATH}")

    try:
        app.state.model = mlflow.pyfunc.load_model(
            str(MODEL_PATH)
        )

        print("Modelo P2 cargado correctamente.")

    except Exception as error:
        print(f"Error al cargar el modelo: {error}")
        raise

    yield

    print("Finalizando API MiBici...")


# ==========================================
# CREACIÓN DE LA API
# ==========================================

app = FastAPI(
    title="MiBici - API de Predicción",
    description=(
        "API REST para predecir alta demanda "
        "en estaciones de MiBici utilizando el modelo P2."
    ),
    version="2.0.0",
    lifespan=lifespan
)


# ==========================================
# ENDPOINT DE SALUD
# ==========================================

@app.get("/health")
def health():

    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=503,
            detail="El modelo no está cargado."
        )

    return {
        "status": "ok",
        "modelo": MODEL_NAME,
        "version": MODEL_VERSION,
        "serving": "disponible"
    }


# ==========================================
# ENDPOINT DE PREDICCIÓN
# ==========================================

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):

    try:

        # Preparar los datos de entrada
        dataframe = pd.DataFrame(
            [[
                data.promedio_duracion_salida,
                data.pct_viajes_matutinos,
                data.viajes_entrantes
            ]],
            columns=FEATURES
        )

        # Ejecutar predicción directamente con el modelo
        model = app.state.model

        result = model.predict(dataframe)

        prediction = int(result[0])

        descripcion = (
            "Alta demanda"
            if prediction == 1
            else "Demanda no alta"
        )

        return PredictionOutput(
            modelo=MODEL_NAME,
            version=MODEL_VERSION,
            prediccion=prediction,
            descripcion=descripcion
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Error al ejecutar la predicción: {str(error)}"
        )
