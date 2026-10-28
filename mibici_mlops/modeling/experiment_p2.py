"""
Experimentación de modelos para P2 — MiBici (Parte 1/2).

Objetivo:
Módulos de carga de datos, partición estratificada 80/20
y configuración de la infraestructura de MLflow Tracking.
"""

import os
import hashlib
import json
import subprocess
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

# CORRECCIÓN SOLUCIÓN: Importar la ruta raíz oficial calculada por tu proyecto
from mibici_mlops.config import PROJ_ROOT

# Variables globales compartidas por convención del proyecto (P2)
QUESTION = "P2"
MODEL_FAMILY = "RandomForestClassifier"

FEATURES = [
    "promedio_duracion_salida",
    "pct_viajes_matutinos",
    "viajes_entrantes",
]
TARGET = "alta_demanda"
RANDOM_STATE = 42

# Asignación limpia utilizando el Path verificado por tu sistema
PROCESSED_DATA_DIR = PROJ_ROOT / "data" / "processed"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mibici-equipo48")


def load_features_data(input_path: Path = PROCESSED_DATA_DIR / "features_estaciones.parquet"):
    """Carga el dataset de variables usando Polars y extrae el hash SHA-256."""
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")
    
    print(f"Leyendo características desde: {input_path}")
    df = pl.read_parquet(input_path)
    
    manifest_path = PROJ_ROOT / "data" / "manifest.json"
    data_version = "unknown"
    if manifest_path.exists():
        data_version = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        
    return df, data_version


def prepare_data():
    """Prepara las matrices de entrenamiento y prueba usando división estratificada."""
    df_polars, data_version = load_features_data()
    required_columns = FEATURES + [TARGET]
    df_clean = df_polars.select(required_columns).drop_nulls()
    
    if df_clean.height == 0:
        raise ValueError("No hay registros disponibles después de eliminar nulos.")
        
    X = df_clean.select(FEATURES).to_pandas()
    y = df_clean.select(TARGET).to_pandas().values.ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print("\n===== PARTICIÓN ESTRATIFICADA P2 =====")
    print(f"Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, data_version


def prepare_validation(X_train, y_train):
    """Sub-divide internamente el conjunto de entrenamiento de forma estratificada."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    print(f"Ajuste Interno: {len(X_fit):,} | Validación Interna: {len(X_val):,}")
    return X_fit, X_val, y_fit, y_val


def configure_mlflow():
    """Establece la URI de almacenamiento relacional local para el Model Registry."""
    mlflow_db = (PROJ_ROOT / "mlflow.db").resolve()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{mlflow_db.as_posix()}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Tracking URI: {tracking_uri} | Experimento: {EXPERIMENT_NAME}")


def build_and_evaluate_model(n_estimators, max_depth, X_train, y_train, X_test, y_test):
    """Entrena el clasificador con pesos balanceados y calcula métricas."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    start_train = time.perf_counter()
    clf.fit(X_train, y_train)
    training_time = time.perf_counter() - start_train
    
    start_infer = time.perf_counter()
    preds = clf.predict(X_test)
    probabilidades = clf.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - start_infer
    
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_f1_score": float(f1_score(y_test, preds, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilidades)),
        "training_time_seconds": float(training_time),
        "inference_time_seconds": float(inference_time)
    }
    return clf, metrics, preds


def get_git_commit():
    """Recupera el hash SHA del commit actual para trazabilidad de código."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_candidate(run_name, n_estimators, max_depth, X_train, y_train, X_test, y_test, data_version, stage_of_work="candidate", nested=False):
    """Centraliza el entrenamiento y empaquetado de artefactos de P2 en MLflow."""
    model, metrics, preds = build_and_evaluate_model(n_estimators, max_depth, X_train, y_train, X_test, y_test)
    
    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.set_tags({
            "question": QUESTION,
            "model_family": MODEL_FAMILY,
            "stage_of_work": stage_of_work,
            "team": EXPERIMENT_NAME,
            "git_commit": get_git_commit(),
            "data_version": data_version
        })
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth, "class_weight": "balanced"})
        
        prefix = "val" if stage_of_work == "search" else "test"
        logged_metrics = {
            (k.replace("test_", f"{prefix}_", 1) if k.startswith("test_") else k): v 
            for k, v in metrics.items()
        }
        mlflow.log_metrics(logged_metrics)
        
        # Guardado de diccionarios de configuración y datos
        mlflow.log_dict({"question": QUESTION, "stage_of_work": stage_of_work, "data_version": data_version}, "data/manifest.json")
        mlflow.log_dict({"run_name": run_name, "hyperparameters": {"n_estimators": n_estimators, "max_depth": max_depth}}, "config/model_config.json")
        
        # Guardar gráfico de Matriz de Confusión
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Alta Demanda (1)"])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/confusion_matrix.png")
        plt.close(fig)
        
        # Log del Modelo con Firma
        input_example = X_train.head(3).copy().astype(np.float32)
        signature = infer_signature(input_example, model.predict(input_example))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"]
        )
        print(f">> Fin [{run_name}] -> F1-Score: {metrics['test_f1_score']:.4f}")
    return metrics


def log_confusion_matrix_plot(y_true, y_pred, run_name):
    """Genera y guarda el gráfico de la matriz de confusión en subcarpetas de MLflow."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Alta Demanda (1)"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)

def search_hyperparameters(X_fit, y_fit, X_val, y_val, data_version):
    """Ejecuta la búsqueda de hiperparámetros estructurada como Nested Runs."""
    configurations = [
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 15},
        {"n_estimators": 200, "max_depth": 20},
    ]
    results = []
    
    with mlflow.start_run(run_name="P2_hyperparameter_search"):
        mlflow.set_tags({"question": QUESTION, "model_family": MODEL_FAMILY, "stage_of_work": "search", "team": EXPERIMENT_NAME, "git_commit": get_git_commit()})
        
        for i, config in enumerate(configurations, start=1):
            metrics = run_candidate(f"P2_search_candidate_{i}", config["n_estimators"], config["max_depth"], X_fit, y_fit, X_val, y_val, data_version, stage_of_work="search", nested=True)
            results.append({"run_name": f"P2_search_candidate_{i}", **config, "val_f1_score": metrics["test_f1_score"]})
            
        best_config = max(results, key=lambda item: item["val_f1_score"])
        mlflow.log_params({"search_strategy": "manual_grid", "selection_metric": "val_f1_score"})
        mlflow.log_metrics({"best_val_f1_score": best_config["val_f1_score"]})
        print(f"\n===== MEJOR CONFIGURACIÓN ENCONTRADA: E={best_config['n_estimators']}, D={best_config['max_depth']} =====")
    return best_config, results


def evaluate_final_candidates(X_train, y_train, X_test, y_test, data_version):
    """Evalúa el modelo baseline original y los tres candidatos sobre el test set."""
    configurations = [
        ("P2_original_baseline", 10, 5, "baseline"),
        ("P2_candidate_1", 50, 10, "candidate"),
        ("P2_candidate_2", 100, 15, "candidate"),
        ("P2_candidate_3", 200, 20, "candidate"),
    ]
    results = []
    for run_name, n_estimators, max_depth, stage in configurations:
        metrics = run_candidate(run_name, n_estimators, max_depth, X_train, y_train, X_test, y_test, data_version, stage_of_work=stage)
        results.append({"model_run": run_name, "F1-Score": metrics["test_f1_score"], "ROC-AUC": metrics["test_roc_auc"]})
    print("\n===== COMPARACIÓN FINAL DE CANDIDATOS P2 =====")
    print(pd.DataFrame(results).to_string(index=False))


def main():
    configure_mlflow()
    X_train, X_test, y_train, y_test, data_version = prepare_data()
    X_fit, X_val, y_fit, y_val = prepare_validation(X_train, y_train)
    best_config, search_results = search_hyperparameters(X_fit, y_fit, X_val, y_val, data_version)
    evaluate_final_candidates(X_train, y_train, X_test, y_test, data_version)
    print("\nEjecución del experimento P2 finalizada correctamente.")


if __name__ == "__main__":
    main()
