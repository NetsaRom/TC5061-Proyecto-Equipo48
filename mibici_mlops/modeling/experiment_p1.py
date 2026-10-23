"""
Experimentación de modelos para P1 — MiBici.

Objetivo:
Comparar modelos de regresión para predecir
la cantidad de viajes de la siguiente hora
por estación.

Parte B — Segunda entrega de MLOps.
"""

import os
import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import subprocess

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mibici_mlops.modeling.train_p1_regression import (
    load_trips,
    build_features,
    ROOT_DIR,
    EXPERIMENT_NAME,
    RANDOM_STATE,
)

# Configuración de experimentación
QUESTION = "P1"
MODEL_FAMILY = "RandomForestRegressor"

FEATURES = [
    "estacion",
    "hora_dia",
    "dia_semana",
    "mes",
    "fin_semana",
    "viajes_lag_1",
    "viajes_lag_24",
    "viajes_lag_168",
    "promedio_24h",
]

TARGET = "target_viajes_siguiente_hora"

def prepare_data():
    """
    Carga los datos de MiBici y realiza una división
    cronológica 80/20, igual que el modelo original.
    """

    trips, file_hashes = load_trips()
    data = build_features(trips)

    # Ordenar las horas disponibles
    timestamps = np.sort(data["hora"].dropna().unique())

    if len(timestamps) < 2:
        raise ValueError(
            "Se necesitan al menos dos horas para dividir los datos."
        )

    # División temporal 80 % entrenamiento, 20 % prueba
    cutoff_idx = int(len(timestamps) * 0.8)
    cutoff_idx = min(max(cutoff_idx, 1), len(timestamps) - 1)
    cutoff = timestamps[cutoff_idx]

    train = data[data["hora"] < cutoff].copy()
    test = data[data["hora"] >= cutoff].copy()

    if train.empty or test.empty:
        raise ValueError("La división temporal produjo un conjunto vacío.")

    # Evitar que el objetivo de la última hora de entrenamiento
    # corresponda a la primera hora del periodo de prueba.
    train = train[
        train["hora"] + pd.Timedelta(hours=1) < pd.Timestamp(cutoff)
    ].copy()

    if train.empty:
        raise ValueError(
            "No quedaron observaciones de entrenamiento "
            "después de aplicar la separación temporal."
        )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("\n===== PARTICIÓN TEMPORAL P1 =====")
    print(f"Fecha de corte: {pd.Timestamp(cutoff)}")
    print(f"Entrenamiento: {len(train):,} observaciones")
    print(f"Prueba: {len(test):,} observaciones")

    train_hours = train["hora"].copy()
    test_hours = test["hora"].copy()

    return X_train, X_test, y_train, y_test, file_hashes, cutoff, train_hours, test_hours

def build_model(n_estimators=100, max_depth=15):
    """
    Construye un pipeline de regresión para P1.

    Permite experimentar con diferentes hiperparámetros
    conservando el mismo preprocesamiento.
    """

    categorical = ["estacion"]
    numeric = [col for col in FEATURES if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "station",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
        ),
    ])

    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Entrena un modelo y evalúa su rendimiento
    utilizando MAE y RMSE.
    """

    import time

    # Medir tiempo de entrenamiento
    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    # Medir tiempo de inferencia
    start_time = time.perf_counter()

    predictions = model.predict(X_test)

    inference_time = time.perf_counter() - start_time

    # Calcular métricas
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    metrics = {
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "training_time_seconds": float(training_time),
        "inference_time_seconds": float(inference_time),
    }

    return metrics, predictions

def configure_mlflow():
    """
    Configura MLflow Tracking para los experimentos P1.
    """

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\n===== CONFIGURACIÓN MLFLOW =====")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experimento: {EXPERIMENT_NAME}")

    return tracking_uri

def get_git_commit():
    """
    Obtiene el identificador del commit actual de Git.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def run_candidate(
    run_name,
    n_estimators,
    max_depth,
    X_train,
    y_train,
    X_test,
    y_test,
    file_hashes,
    cutoff,
    stage_of_work="candidate",
    nested=False,
):
    """
    Entrena y registra un candidato de P1 en MLflow.
    """

    model = build_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    with mlflow.start_run(
        run_name=run_name,
        nested=nested,
    ):
        # Identificación del experimento
        mlflow.set_tags({
            "question": QUESTION,
            "model_family": MODEL_FAMILY,
            "stage_of_work": stage_of_work,
            "team": EXPERIMENT_NAME,
            "git_commit": get_git_commit(),
        })

        # Hiperparámetros
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": RANDOM_STATE,
        })

        # Entrenamiento y evaluación
        metrics, predictions = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        metric_prefix = (
            "val" if stage_of_work == "search" else "test"
        )

        logged_metrics = {
            (
                key.replace("test_", f"{metric_prefix}_", 1)
                if key.startswith("test_")
                else key
            ): value
            for key, value in metrics.items()
        }

        mlflow.log_metrics(logged_metrics)

        # Registrar trazabilidad de los datos
        log_data_manifest(
            file_hashes=file_hashes,
            cutoff=cutoff,
        )

        # Registrar configuración del candidato
        log_candidate_config(
            run_name=run_name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            stage_of_work=stage_of_work,
        )

        # Registrar gráfica de predicciones
        log_prediction_plot(
            y_true=y_test,
            predictions=predictions,
            run_name=run_name,
        )

        # Registrar el modelo completo en MLflow
        input_example = X_train.head(5).copy()

        # Obtener predicciones de ejemplo con el modelo entrenado
        example_predictions = model.predict(input_example)

        # Inferir la estructura de entradas y salidas
        signature = infer_signature(
            input_example,
            example_predictions,
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        print(f"\n===== {run_name} =====")
        print(f"MAE: {metrics['test_mae']:.4f}")
        print(f"RMSE: {metrics['test_rmse']:.4f}")
        print(
            f"Entrenamiento: "
            f"{metrics['training_time_seconds']:.2f} s"
        )

    return metrics

def prepare_validation(X_train, y_train, train_hours):
    """
    Divide cronológicamente los datos de entrenamiento:
    80 % para ajuste y 20 % para validación.

    No vuelve a cargar ni procesar los datos originales.
    """

    # Alinear las fechas con las observaciones
    train_hours = train_hours.loc[X_train.index]

    # Identificar las horas disponibles
    timestamps = np.sort(train_hours.dropna().unique())

    if len(timestamps) < 3:
        raise ValueError(
            "No hay suficientes horas para crear validación temporal."
        )

    # Definir el corte cronológico
    cutoff_idx = int(len(timestamps) * 0.8)
    cutoff_idx = min(max(cutoff_idx, 1), len(timestamps) - 1)
    validation_cutoff = pd.Timestamp(timestamps[cutoff_idx])

    # Evitar que los objetivos de entrenamiento
    # correspondan al periodo de validación
    fit_mask = (
        train_hours + pd.Timedelta(hours=1)
        < validation_cutoff
    )

    validation_mask = train_hours >= validation_cutoff

    # Crear los conjuntos
    X_fit = X_train.loc[fit_mask]
    y_fit = y_train.loc[fit_mask]

    X_val = X_train.loc[validation_mask]
    y_val = y_train.loc[validation_mask]

    if X_fit.empty or X_val.empty:
        raise ValueError(
            "La partición temporal produjo un conjunto vacío."
        )

    print("\n===== VALIDACIÓN TEMPORAL P1 =====")
    print(f"Corte de validación: {validation_cutoff}")
    print(f"Entrenamiento interno: {len(X_fit):,}")
    print(f"Validación: {len(X_val):,}")

    return X_fit, X_val, y_fit, y_val, validation_cutoff

def log_data_manifest(
    file_hashes,
    cutoff,
    validation_cutoff=None,
):
    """
    Registra la trazabilidad de los datos y
    las fechas de separación temporal en MLflow.
    """

    manifest = {
        "question": QUESTION,
        "dataset": "MiBici Guadalajara",
        "split_strategy": "chronological",
        "test_cutoff": str(pd.Timestamp(cutoff)),
        "validation_cutoff": (
            str(pd.Timestamp(validation_cutoff))
            if validation_cutoff is not None
            else None
        ),
        "files_sha256": file_hashes,
    }

    mlflow.log_dict(
        manifest,
        "data/manifest.json",
    )

def log_candidate_config(
    run_name,
    n_estimators,
    max_depth,
    stage_of_work,
):
    """
    Registra la configuración del candidato
    como artefacto de MLflow.
    """

    config = {
        "question": QUESTION,
        "run_name": run_name,
        "model_family": MODEL_FAMILY,
        "stage_of_work": stage_of_work,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": RANDOM_STATE,
        },
        "features": FEATURES,
        "target": TARGET,
        "evaluation_metrics": [
            "test_mae",
            "test_rmse",
        ],
    }

    mlflow.log_dict(
        config,
        "config/model_config.json",
    )

def log_prediction_plot(y_true, predictions, run_name):
    """
    Genera y registra en MLflow una gráfica
    de valores reales frente a predichos.
    """

    # Limitar la visualización a 300 observaciones
    n = min(300, len(y_true))

    real = np.asarray(y_true)[:n]
    predicted = np.asarray(predictions)[:n]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        range(n),
        real,
        label="Viajes reales",
        alpha=0.8,
    )

    ax.plot(
        range(n),
        predicted,
        label="Viajes predichos",
        alpha=0.8,
    )

    ax.set_title(f"P1: Reales vs. predichos - {run_name}")
    ax.set_xlabel("Observación")
    ax.set_ylabel("Número de viajes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Registrar directamente la figura en MLflow
    mlflow.log_figure(
        fig,
        "plots/actual_vs_predicted.png",
    )

    plt.close(fig)

def search_hyperparameters(
    X_fit,
    y_fit,
    X_val,
    y_val,
    file_hashes,
    cutoff,
):
    """
    Ejecuta una búsqueda de hiperparámetros
    utilizando corridas anidadas de MLflow.
    """

    configurations = [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
        {"n_estimators": 300, "max_depth": 20},
    ]

    results = []

    # Corrida padre: representa la búsqueda completa
    with mlflow.start_run(
        run_name="P1_hyperparameter_search"
    ):
        mlflow.set_tags({
            "question": QUESTION,
            "model_family": MODEL_FAMILY,
            "stage_of_work": "search",
            "team": EXPERIMENT_NAME,
            "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        })

        for i, config in enumerate(configurations, start=1):
            run_name = f"P1_search_candidate_{i}"

            # Cada combinación crea una corrida hija
            metrics = run_candidate(
                run_name=run_name,
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                X_train=X_fit,
                y_train=y_fit,
                X_test=X_val,
                y_test=y_val,
                file_hashes=file_hashes,
                cutoff=cutoff,
                stage_of_work="search",
                nested=True,
            )

            results.append({
                "run_name": run_name,
                "n_estimators": config["n_estimators"],
                "max_depth": config["max_depth"],
                "val_mae": metrics["test_mae"],
                "val_rmse": metrics["test_rmse"],
            })

        best = min(results, key=lambda item: item["val_mae"])

        mlflow.log_params({
            "search_strategy": "manual_grid",
            "n_combinations": len(configurations),
            "selection_metric": "val_mae",
        })

        mlflow.log_metrics({
            "best_val_mae": best["val_mae"],
            "best_val_rmse": best["val_rmse"],
        })

        print("\n===== MEJOR CONFIGURACIÓN P1 =====")
        print(f"Modelo: {best['run_name']}")
        print(f"Árboles: {best['n_estimators']}")
        print(f"Profundidad: {best['max_depth']}")
        print(f"MAE validación: {best['val_mae']:.4f}")

    return best, results

def main():
    """
    Ejecuta la preparación de datos y la búsqueda
    de hiperparámetros para P1.
    """

    configure_mlflow()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        file_hashes,
        cutoff,
        train_hours,
        test_hours,
    ) = prepare_data()

    X_fit, X_val, y_fit, y_val, validation_cutoff = prepare_validation(
    X_train,
    y_train,
    train_hours,
    )

    best, results = search_hyperparameters(
        X_fit=X_fit,
        y_fit=y_fit,
        X_val=X_val,
        y_val=y_val,
        file_hashes=file_hashes,
        cutoff=cutoff,
    )

    print("\n===== RESUMEN DE BÚSQUEDA P1 =====")
    for result in results:
        print(
            f"{result['run_name']}: "
            f"MAE={result['val_mae']:.4f}, "
            f"RMSE={result['val_rmse']:.4f}"
        )

    print("\nMejor configuración:")
    print(best)

    # Evaluación final: modelo original y tres candidatos
    comparison = evaluate_final_candidates(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        file_hashes=file_hashes,
        cutoff=cutoff,
    )

    # Identificar la configuración recomendada por validación
    print("\n===== SELECCIÓN POR VALIDACIÓN =====")
    print(f"Árboles: {best['n_estimators']}")
    print(f"Profundidad: {best['max_depth']}")
    print(f"MAE validación: {best['val_mae']:.4f}")

    print("\nEvaluación final completada.")

def evaluate_final_candidates(
    X_train,
    y_train,
    X_test,
    y_test,
    file_hashes,
    cutoff,
):
    """
    Evalúa el modelo original y tres candidatos
    utilizando el mismo conjunto de prueba.
    """

    configurations = [
        ("P1_original", 100, 15, "baseline"),
        ("P1_candidate_1", 100, 10, "candidate"),
        ("P1_candidate_2", 200, 15, "candidate"),
        ("P1_candidate_3", 300, 20, "candidate"),
    ]

    results = []

    for run_name, n_estimators, max_depth, stage in configurations:

        metrics = run_candidate(
            run_name=run_name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            file_hashes=file_hashes,
            cutoff=cutoff,
            stage_of_work=stage,
        )

        results.append({
            "model": run_name,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            **metrics,
        })

    comparison = pd.DataFrame(results)

    print("\n===== COMPARACIÓN FINAL P1 =====")
    print(comparison.to_string(index=False))

    return comparison

if __name__ == "__main__":
    main()