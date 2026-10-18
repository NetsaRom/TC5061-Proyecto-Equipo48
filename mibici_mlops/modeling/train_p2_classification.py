
from pathlib import Path

import typer
import polars as pl
import mlflow
import mlflow.sklearn

from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from mibici_mlops.config import PROCESSED_DATA_DIR, PROJ_ROOT


app = typer.Typer()

FEATURES = [
    "promedio_duracion_salida",
    "pct_viajes_matutinos",
    "viajes_entrantes",
]

TARGET = "alta_demanda"


@app.command()
def entrenar_clasificador(
    input_path: Path = PROCESSED_DATA_DIR / "features_estaciones.parquet",
    n_estimators: int = typer.Option(100, "--estimators", "-e"),
    test_size: float = typer.Option(0.2, "--test-size"),
):
    """Entrena y evalúa un clasificador Random Forest para estaciones MiBici."""

    # Validaciones iniciales
    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: {input_path}"
        )

    if n_estimators < 1:
        raise ValueError("n_estimators debe ser mayor que cero.")

    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")

    logger.info(f"Leyendo datos desde: {input_path}")
    df = pl.read_parquet(input_path)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas en el dataset: {missing_columns}"
        )

    # Selección de características y variable objetivo
    df = df.select(required_columns).drop_nulls()

    if df.height == 0:
        raise ValueError("No hay registros disponibles después de eliminar nulos.")

    X = df.select(FEATURES).to_numpy()
    y = df[TARGET].to_numpy()

    classes = set(y.tolist())

    if classes != {0, 1}:
        raise ValueError(
            f"La variable '{TARGET}' debe contener las clases 0 y 1. "
            f"Clases encontradas: {classes}"
        )

    if min(sum(y == 0), sum(y == 1)) < 2:
        raise ValueError(
            "Se necesitan al menos dos registros de cada clase "
            "para realizar la división estratificada."
        )

    # División estratificada para conservar la proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    # Configuración de MLflow con SQLite
    mlflow_db = (PROJ_ROOT / "mlflow.db").resolve()
    tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("MiBici_Clasificacion_Estaciones")

    with mlflow.start_run(
        run_name=f"RandomForest_Clasif_E{n_estimators}"
    ):
        logger.info("Entrenando modelo Random Forest Classifier...")

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        clf.fit(X_train, y_train)

        # Predicciones
        preds = clf.predict(X_test)
        probabilidades = clf.predict_proba(X_test)[:, 1]

        # Métricas
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probabilidades)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1-score: {f1:.4f}")
        logger.info(f"ROC-AUC: {auc:.4f}")

        logger.info(
            "Matriz de confusión:\n"
            f"{confusion_matrix(y_test, preds, labels=[0, 1])}"
        )

        logger.info(
            "Reporte de clasificación:\n"
            f"{classification_report(y_test, preds, zero_division=0)}"
        )

        # Registro de parámetros y métricas
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", clf.max_depth)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("target", TARGET)
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(
            sk_model=clf,
            name="modelo_clasificador",
            input_example=X_train[:5],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        logger.success("Modelo de clasificación registrado en MLflow.")


if __name__ == "__main__":
    app()
