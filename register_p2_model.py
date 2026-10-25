"""
Registro y gobernanza de modelos en el Model Registry — MiBici P2.
Ubicación: Raíz del repositorio.

Objetivo:
Registrar el mejor modelo candidato de clasificación en el servidor 
local y asignarle el alias lógico '@champion' para producción.
"""

import sys
from pathlib import Path
import mlflow
from mlflow import MlflowClient

# AJUSTE DE SEGURIDAD: Asegurar que Python encuentre el módulo 'mibici_mlops' 
# si se ejecuta el script de manera directa (ej. 'python register_p2_model.py')
RAIZ_PROYECTO = Path(__file__).resolve().parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

# Configuración explícita de URIs para base de datos SQLite local en la raíz
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Nombre oficial del contenedor en el Registry para el clasificador de riesgo
MODEL_NAME = "mibici-equipo48-p2-riesgo"

# ID real verificado de tu corrida candidata de clasificación exitosa
RUN_ID = "a325a202aa4447158aec1ce49bdf69d0"

# El artifact_path del log_model original en el script de clasificación se llama "model"
model_uri = f"runs:/{RUN_ID}/model"

# 1. Registrar el modelo en el contenedor centralizado
model_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

version = model_version.version

# 2. Establecer la descripción general del propósito del contenedor
client.update_registered_model(
    name=MODEL_NAME,
    description=(
        "Modelo de clasificación supervisada para predecir si una "
        "estación de MiBici Guadalajara experimentará un estado de alta "
        "demanda en la siguiente hora, anticipando desabasto operativo."
    ),
)

# 3. Establecer la descripción específica de esta versión (Linaje e hiperparámetros)
client.update_model_version(
    name=MODEL_NAME,
    version=version,
    description=(
        "Modelo procedente de la corrida P2_candidate_2. Random Forest "
        "Classifier con n_estimators=100, max_depth=15 y class_weight='balanced'. "
        "F1-Score en prueba de 0.9782, ROC-AUC de 0.9990. Revisar metadatos asociados."
    ),
)

# 4. Asignar el alias lógico moderno exigido para producción
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=version,
)

# 5. Prueba de humo (Smoke Test) para verificar la carga por alias desde la URI lógica
loaded_model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@champion"
)

print(f"Modelo registrado: {MODEL_NAME}")
print(f"Versión: {version}")
print("Alias: champion")
print("Modelo cargado correctamente desde la raíz del proyecto.")
