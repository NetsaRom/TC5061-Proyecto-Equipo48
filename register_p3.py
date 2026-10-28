
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

client = MlflowClient()

MODEL_NAME = "mibici-equipo48-p3-grupos"

# ID de la corrida
RUN_ID = "2266d8347031427397261cdd27ceff51"

model_uri = f"runs:/{RUN_ID}/modelo_kmeans"


model_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

version = model_version.version

client.update_registered_model(
    name=MODEL_NAME,
    description=(
        "Modelo K-Means para agrupar estaciones de MiBici Guadalajara "
        "según viajes salientes, duración promedio y proporción de "
        "viajes matutinos. La selección debe considerar las métricas "
        "de clustering y la validación del proyecto."
    ),
)

client.update_model_version(
    name=MODEL_NAME,
    version=version,
    description=(
        "Modelo procedente de la corrida P3_candidate_k3. "
        "K-Means con 3 clústeres. Revisar métricas, commit y "
        "versión de datos de la corrida de origen."
    ),
)

# Asignar alias a la versión registrada
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=version,
)

# Probar que se pueda cargar por alias
loaded_model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@champion"
)

print(f"Modelo registrado: {MODEL_NAME}")
print(f"Versión: {version}")
print("Alias: champion")
print("Modelo cargado correctamente.")
