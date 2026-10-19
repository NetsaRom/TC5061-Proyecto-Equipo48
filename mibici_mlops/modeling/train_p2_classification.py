
from pathlib import Path
import subprocess

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
EXPERIMENT_NAME = "mibici-equipo48"
TEAM_NAME = "mibici-equipo48"


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
        raise ValueError(
            "No hay registros disponibles después de eliminar nulos."
        )

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

    # División estratificada
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
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Obtener el commit actual del repositorio
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning(f"No fue posible obtener el commit de Git: {exc}")
        git_commit = "unknown"

    # Obtener versión de datos desde el manifiesto, si existe
    manifest_path = PROJ_ROOT / "data" / "manifest.json"
    data_version = "unknown"

    if manifest_path.exists():
        import hashlib

        data_version = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()

    with mlflow.start_run(
        run_name=f"P2_random_forest_clasificacion_E{n_estimators}"
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

        # Matriz de confusión y reporte
        matriz = confusion_matrix(y_test, preds, labels=[0, 1])
        reporte = classification_report(
            y_test,
            preds,
            labels=[0, 1],
            zero_division=0,
        )

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1-score: {f1:.4f}")
        logger.info(f"ROC-AUC: {auc:.4f}")
        logger.info(f"Matriz de confusión:\n{matriz}")
        logger.info(f"Reporte de clasificación:\n{reporte}")

        # Parámetros
        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "max_depth": str(clf.max_depth),
            "random_state": 42,
            "test_size": test_size,
            "features": ",".join(FEATURES),
            "target": TARGET,
            "class_weight": "balanced",
            "dataset_path": str(input_path),
        })

        # Métricas
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc,
        })

        # Etiquetas de trazabilidad
        mlflow.set_tags({
            "team": TEAM_NAME,
            "question": "P2",
            "git_commit": git_commit,
            "data_version": data_version,
            "target_type": "synthetic_high_demand_proxy",
        })

        # Artefactos de evaluación
        artifacts_dir = PROJ_ROOT / "reports" / "figures"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        report_path = artifacts_dir / "p2_classification_report.txt"
        report_path.write_text(
            "MiBici - P2: Clasificación de alta demanda\n\n"
            f"Experimento: {EXPERIMENT_NAME}\n"
            f"Commit: {git_commit}\n"
            f"Versión del manifiesto: {data_version}\n"
            f"Accuracy: {acc:.4f}\n"
            f"F1-score: {f1:.4f}\n"
            f"ROC-AUC: {auc:.4f}\n\n"
            "Matriz de confusión (clases 0, 1):\n"
            f"{matriz}\n\n"
            "Reporte de clasificación:\n"
            f"{reporte}\n\n"
            "Limitación: la etiqueta alta_demanda es un proxy sintético "
            "basado en la demanda observada; no representa mediciones "
            "reales de inventario ni confirma desabasto de bicicletas.\n",
            encoding="utf-8",
        )

        mlflow.log_artifact(
            str(report_path),
            artifact_path="evaluacion",
        )

        # Registro del modelo
        mlflow.sklearn.log_model(
            sk_model=clf,
            name="modelo_clasificador",
            input_example=X_train[:5],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        logger.success(
            "Modelo, métricas, etiquetas y reporte registrados en MLflow."
        )


if __name__ == "__main__":
    app()
