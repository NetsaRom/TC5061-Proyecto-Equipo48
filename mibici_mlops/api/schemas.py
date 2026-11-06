from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Variables necesarias para predecir los viajes de la siguiente hora."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    estacion: Annotated[str, Field(min_length=1)]
    hora_dia: Annotated[int, Field(ge=0, le=23)]
    dia_semana: Annotated[int, Field(ge=0, le=6)]
    mes: Annotated[int, Field(ge=1, le=12)]
    fin_semana: Annotated[int, Field(ge=0, le=1)]

    viajes_lag_1: Annotated[float, Field(ge=0)]
    viajes_lag_24: Annotated[float, Field(ge=0)]
    viajes_lag_168: Annotated[float, Field(ge=0)]
    promedio_24h: Annotated[float, Field(ge=0)]


class PredictionResponse(BaseModel):
    """Resultado de la predicción y metadatos del modelo."""

    prediction: float
    unit: str = "viajes"
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Estado de disponibilidad de la API y del modelo."""

    status: str
    model_loaded: bool
    model_name: str
    model_version: str
