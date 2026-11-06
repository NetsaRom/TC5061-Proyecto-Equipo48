import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_path = mlflow.artifacts.download_artifacts(
    artifact_uri="models:/mibici-equipo48-p2-riesgo/1",
    dst_path="./models/p2"
)

print("Modelo descargado en:", model_path)