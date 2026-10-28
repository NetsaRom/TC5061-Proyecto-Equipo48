
"""
Registro y gobernanza del modelo P2 — MiBici MLOps.
Ubicación: raíz del repositorio.

Registra el candidato 3 verificado y asigna el alias champion.
"""

import os
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

# Raíz del repositorio
RAIZ_PROYECTO = Path(__file__).resolve().parent

if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

# Utiliza el mismo servidor de MLflow que el entrenamiento
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

client = MlflowClient()

MODEL_NAME = "mibici-equipo48-p2-riesgo"
RUN_ID = "0766e362faf644129dd54d9a2993380c"
MODEL_ARTIFACT = "model"
ALIAS = "champion"

# 1. Validar la corrida y sus etiquetas
run = client.get_run(RUN_ID)

if run.info.status != "FINISHED":
    raise RuntimeError(
        f"La corrida {RUN_ID} no terminó correctamente."
    )

if run.data.tags.get("question") != "P2":
    raise ValueError(
        f"La corrida {RUN_ID} no está identificada como P2."
    )

if run.data.tags.get("stage_of_work") != "search":
    raise ValueError(
        "La corrida no está identificada como candidata de búsqueda."
    )

# 2. Comprobar que el artefacto del modelo existe
artifacts = client.list_artifacts(RUN_ID, MODEL_ARTIFACT)

if not any(
    artifact.path == f"{MODEL_ARTIFACT}/MLmodel"
    for artifact in artifacts
):
    raise FileNotFoundError(
        f"No se encontró {MODEL_ARTIFACT}/MLmodel "
        f"en la corrida {RUN_ID}."
    )

model_uri = f"runs:/{RUN_ID}/{MODEL_ARTIFACT}"

# 3. Registrar el modelo en el Model Registry
# Si se vuelve a ejecutar, no crear otra versión idéntica.
existing_versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

matching_version = next(
    (
        version
        for version in existing_versions
        if version.run_id == RUN_ID
    ),
    None,
)

if matching_version is None:
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )
    version_number = registered.version
else:
    version_number = matching_version.version

# 4. Describir el modelo sin inventar resultados de prueba
client.update_registered_model(
    name=MODEL_NAME,
    description=(
        "Clasificador Random Forest para la etiqueta alta_demanda "
        "del proyecto MiBici MLOps, equipo 48. La etiqueta es un "
        "proxy de alta demanda y no una medición real de desabasto."
    ),
)

client.update_model_version(
    name=MODEL_NAME,
    version=version_number,
    description=(
        "Procede de la corrida P2_search_candidate_3. "
        f"Run ID: {RUN_ID}. "
        "Consultar los parámetros y las métricas registrados "
        "en MLflow para verificar su configuración y desempeño."
    ),
)

# 5. Asignar el alias champion a esta versión
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias=ALIAS,
    version=version_number,
)

# 6. Prueba de carga del modelo registrado
loaded_model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@{ALIAS}"
)

print(f"Modelo registrado: {MODEL_NAME}")
print(f"Versión: {version_number}")
print(f"Alias: {ALIAS}")
print(f"Run ID: {RUN_ID}")
print("Carga del modelo desde el alias: correcta.")
