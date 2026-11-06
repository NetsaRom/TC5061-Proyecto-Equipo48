import os

import mlflow
import numpy as np
import pandas as pd

MODEL_NAME = "mibici-equipo48-p1-demanda"
MODEL_VERSION = "1"
MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    f"models:/{MODEL_NAME}/{MODEL_VERSION}",
)


class ModelService:
    """Servicio responsable de cargar y ejecutar el modelo de MLflow."""

    def __init__(self):
        self.model = None

    def load(self):
        """Carga el modelo registrado en MLflow."""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            from mibici_mlops.config import PROJ_ROOT

            mlflow.set_tracking_uri(f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")

        self.model = mlflow.pyfunc.load_model(MODEL_URI)
        return self.model

    @property
    def is_loaded(self):
        """Indica si el modelo está disponible en memoria."""
        return self.model is not None

    def predict(self, features: dict) -> float:
        """Ejecuta una predicción con los tipos esperados por MLflow."""
        if not self.is_loaded:
            raise RuntimeError("El modelo no está cargado.")

        data = pd.DataFrame(
            [
                {
                    "estacion": str(features["estacion"]),
                    "hora_dia": np.int32(features["hora_dia"]),
                    "dia_semana": np.int32(features["dia_semana"]),
                    "mes": np.int32(features["mes"]),
                    "fin_semana": np.int64(features["fin_semana"]),
                    "viajes_lag_1": np.float64(features["viajes_lag_1"]),
                    "viajes_lag_24": np.float64(features["viajes_lag_24"]),
                    "viajes_lag_168": np.float64(features["viajes_lag_168"]),
                    "promedio_24h": np.float64(features["promedio_24h"]),
                }
            ]
        )

        prediction = self.model.predict(data)

        return float(prediction[0])
