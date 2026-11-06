from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from mibici_mlops.api.model import MODEL_NAME, MODEL_VERSION, ModelService
from mibici_mlops.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida del modelo."""
    try:
        model_service.load()
    except Exception:  # noqa: BLE001
        model_service.model = None

    yield

    model_service.model = None


app = FastAPI(
    title="MiBici Demand Prediction API",
    description=(
        "API para predecir la demanda de viajes de MiBici "
        "utilizando un modelo de Machine Learning registrado en MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def health():
    """Consulta el estado de disponibilidad de la API y del modelo."""
    response = HealthResponse(
        status="ok" if model_service.is_loaded else "model_unavailable",
        model_loaded=model_service.is_loaded,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )

    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )

    return response


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
)
def predict(payload: PredictionRequest):
    """Predice el número de viajes para la siguiente hora."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de predicción no está disponible.",
        )

    try:
        prediction = model_service.predict(payload.model_dump())

        return PredictionResponse(
            prediction=prediction,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al ejecutar la predicción.",
        ) from exc
