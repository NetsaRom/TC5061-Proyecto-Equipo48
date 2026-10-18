from pathlib import Path
import typer
import polars as pl
import mlflow
import mlflow.sklearn
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mibici_mlops.config import PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()

@app.command()
def entrenar_clasificador(
    input_path: Path = PROCESSED_DATA_DIR / "features_estaciones.parquet",
    n_estimators: int = typer.Option(100, "--estimators", "-e"),
):
    """Entrena un clasificador RandomForest para predecir alta demanda en estaciones."""
    mlflow.set_tracking_uri("file://" + str(PROJ_ROOT / "mlruns"))
    mlflow.set_experiment("MiBici_Clasificacion_Estaciones")

    df = pl.read_parquet(input_path)
    
    # Separar variables y objetivo
    X = df.select(["promedio_duracion_salida", "pct_viajes_matutinos", "viajes_entrantes"]).to_numpy()
    y = df["alta_demanda"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name=f"RandomForest_Clasif_E{n_estimators}"):
        logger.info("Entrenando modelo de clasificación...")
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        probabilidades = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probabilidades)

        logger.info(f"Evaluación Clasificación -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | AUC: {auc:.4f}")

        # MLflow Logs
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(clf, "modelo_clasificador")
        
        logger.success("¡Modelo de clasificación registrado en MLflow!")

if __name__ == "__main__":
    app()
