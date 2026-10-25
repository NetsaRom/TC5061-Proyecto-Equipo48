"""
Experimentación P3: segmentación de estaciones de MiBici con K-Means.

Incluye:
- Baseline con K=4 (valor de N_CLUSTERS configurado en el proyecto).
- Búsqueda de hiperparámetros con runs anidados.
- Métricas de clustering: Silhouette, Davies-Bouldin e inertia.
- Registro de modelos, firma, ejemplo de entrada, manifiesto, configuración y gráficas.
"""

import hashlib
import json
import os
import re
import time
import unicodedata
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici_mlops.config import (
    FECHA_FIN_EXCLUSIVA,
    FECHA_INICIO,
    N_CLUSTERS,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------
# 1. Configuración
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports" / "figures"
CONFIG_DIR = ROOT_DIR / "reports" / "config"

START_DATE = f"{FECHA_INICIO}-01"
END_DATE = FECHA_FIN_EXCLUSIVA
RANDOM_STATE = RANDOM_SEED

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mibici-equipo48")
MODEL_NAME = "KMeans"
FEATURES = [
    "viajes_salientes",
    "promedio_duracion_salida",
    "pct_viajes_matutinos",
]

# El baseline conserva el número de clusters configurado en el proyecto.
BASELINE_K = int(N_CLUSTERS)

# Tres candidatos adicionales al baseline. Se varía el hiperparámetro K.
CANDIDATE_CONFIGS = [
    {"n_clusters": 3, "n_init": 10, "max_iter": 300},
    {"n_clusters": 5, "n_init": 10, "max_iter": 300},
    {"n_clusters": 6, "n_init": 10, "max_iter": 300},
]

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Normalización y detección de columnas
# ---------------------------------------------------------------------

def normalize_column(name):
    """Normaliza nombres para facilitar la detección de columnas."""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def find_column(df, candidates):
    """Encuentra una columna usando nombres alternativos."""
    normalized = {normalize_column(col): col for col in df.columns}
    for candidate in candidates:
        key = normalize_column(candidate)
        if key in normalized:
            return normalized[key]
    return None


# ---------------------------------------------------------------------
# 3. Carga y filtrado de datos
# ---------------------------------------------------------------------

def load_trips():
    """Carga archivos mensuales, normaliza columnas y filtra el periodo."""
    files = sorted(PROCESSED_DIR.glob("viajes_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos Parquet en {PROCESSED_DIR}. "
            "Se esperaban archivos viajes_AAAA_MM.parquet."
        )

    frames = []
    hashes = {}

    for file in files:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        hashes[file.name] = digest
        df = pd.read_parquet(file)

        date_col = find_column(
            df,
            [
                "Inicio_del_viaje", "Fecha_inicio", "fecha_hora_inicio",
                "start_time", "started_at",
            ],
        )
        origin_col = find_column(
            df,
            [
                "Origen_Id", "Estacion_de_Origen", "Estacion_origen",
                "id_estacion_origen", "start_station_id",
            ],
        )
        duration_col = find_column(
            df, ["duracion_min", "duration_minutes", "trip_duration_min"]
        )
        anomaly_col = find_column(
            df, ["duracion_atipica", "is_anomaly", "anomaly"]
        )

        if date_col is None or origin_col is None or duration_col is None:
            raise ValueError(
                f"No se detectaron las columnas necesarias en {file.name}.\n"
                f"Columnas disponibles: {list(df.columns)}"
            )

        if anomaly_col is not None:
            # La columna puede venir como booleano o como 0/1.
            anomaly_values = df[anomaly_col].astype(str).str.lower().str.strip()
            df = df[anomaly_values.isin(["false", "0", "no", "n", ""])]

        part = pd.DataFrame(
            {
                "fecha_inicio": pd.to_datetime(
                    df[date_col], errors="coerce", dayfirst=True
                ),
                "estacion": df[origin_col].astype("string").str.strip(),
                "duracion_min": pd.to_numeric(df[duration_col], errors="coerce"),
            }
        )
        part = part.dropna(subset=["fecha_inicio", "estacion", "duracion_min"])
        part = part[part["estacion"].ne("") & part["estacion"].ne("nan")]
        frames.append(part)

    trips = pd.concat(frames, ignore_index=True)
    trips = trips[
        (trips["fecha_inicio"] >= pd.Timestamp(START_DATE))
        & (trips["fecha_inicio"] < pd.Timestamp(END_DATE))
    ].copy()

    if trips.empty:
        raise ValueError("No hay viajes en el rango de fechas configurado.")

    return trips, hashes


# ---------------------------------------------------------------------
# 4. Ingeniería de características
# ---------------------------------------------------------------------

def build_features(trips):
    """Construye una fila por estación con métricas de comportamiento."""
    trips = trips.copy()
    trips["hora_dia"] = trips["fecha_inicio"].dt.hour

    grouped = trips.groupby("estacion")
    total_viajes = grouped.size().rename("viajes_salientes")
    promedio_duracion = (
        grouped["duracion_min"].mean().rename("promedio_duracion_salida")
    )

    # Hora pico matutina: de 6:00 a 9:59.
    viajes_matutinos = (
        trips[trips["hora_dia"].between(6, 9)].groupby("estacion").size()
    )
    pct_matutino = (
        viajes_matutinos / total_viajes
    ).fillna(0).rename("pct_viajes_matutinos")

    data = pd.concat(
        [total_viajes, promedio_duracion, pct_matutino], axis=1
    ).reset_index()
    data = data.dropna(subset=FEATURES).reset_index(drop=True)

    if len(data) < 3:
        raise ValueError(
            f"Solo hay {len(data)} estaciones con datos; no es suficiente "
            "para comparar los candidatos de clustering."
        )

    print(f"Estaciones identificadas para segmentación: {len(data):,}")
    return data


# ---------------------------------------------------------------------
# 5. Utilidades de MLflow y artefactos
# ---------------------------------------------------------------------

def configure_mlflow():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri


def get_git_commit():
    return os.getenv("GIT_COMMIT", "unknown")


def get_data_version(file_hashes):
    combined = "".join(
        f"{name}:{digest}" for name, digest in sorted(file_hashes.items())
    )
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def log_data_manifest(file_hashes, data_version):
    manifest_path = ROOT_DIR / "data" / "manifest.json"
    if manifest_path.is_file():
        mlflow.log_artifact(str(manifest_path), artifact_path="data")
    else:
        # Se registra un manifiesto mínimo si el repositorio no tiene uno.
        generated_manifest = CONFIG_DIR / "p3_data_manifest.json"
        generated_manifest.write_text(
            json.dumps(
                {
                    "question": "P3",
                    "start_date": START_DATE,
                    "end_date_exclusive": END_DATE,
                    "data_version_sha256": data_version,
                    "source_files_sha256": file_hashes,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(generated_manifest), artifact_path="data")


def log_candidate_config(config, stage, run_name):
    config_path = CONFIG_DIR / f"{run_name}_config.json"
    config_path.write_text(
        json.dumps(
            {
                "question": "P3",
                "stage_of_work": stage,
                "algorithm": MODEL_NAME,
                "features": FEATURES,
                "start_date": START_DATE,
                "end_date_exclusive": END_DATE,
                "random_state": RANDOM_STATE,
                **config,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(config_path), artifact_path="config")


def log_cluster_plot(data, run_name, title):
    """Guarda el perfil visual de los clusters."""
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        data["viajes_salientes"],
        data["promedio_duracion_salida"],
        c=data["cluster"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="white",
        s=55,
    )
    ax.set_title(title)
    ax.set_xlabel("Viajes salientes por estación")
    ax.set_ylabel("Duración promedio de los viajes (min)")
    ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    figure_path = REPORTS_DIR / f"{run_name}_clusters.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(figure_path), artifact_path="plots")


# ---------------------------------------------------------------------
# 6. Entrenar y evaluar un candidato
# ---------------------------------------------------------------------

def evaluate_candidate(X, data, config, run_name, stage, file_hashes,
                       data_version, git_commit):
    """Entrena un K-Means y registra métricas, modelo y artefactos."""
    n_clusters = int(config["n_clusters"])
    if n_clusters >= len(data):
        raise ValueError(
            f"K={n_clusters} debe ser menor que el número de estaciones "
            f"({len(data)})."
        )

    model = make_pipeline(
        StandardScaler(),
        KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=int(config.get("n_init", 10)),
            max_iter=int(config.get("max_iter", 300)),
        ),
    )

    start = time.perf_counter()
    labels = model.fit_predict(X)
    training_seconds = time.perf_counter() - start

    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        raise ValueError(
            f"K={n_clusters} produjo {len(unique_labels)} clusters válidos; "
            "no se pueden calcular las métricas de clustering."
        )

    X_scaled = model.named_steps["standardscaler"].transform(X)
    silhouette = silhouette_score(X_scaled, labels)
    db_index = davies_bouldin_score(X_scaled, labels)
    inertia = float(model.named_steps["kmeans"].inertia_)

    mlflow.set_tags(
        {
            "team": EXPERIMENT_NAME,
            "question": "P3",
            "model_family": "clustering",
            "stage_of_work": stage,
            "git_commit": git_commit,
            "data_version": data_version,
        }
    )
    mlflow.log_params(
        {
            "algorithm": MODEL_NAME,
            "n_clusters": n_clusters,
            "n_init": int(config.get("n_init", 10)),
            "max_iter": int(config.get("max_iter", 300)),
            "random_state": RANDOM_STATE,
            "start_date": START_DATE,
            "end_date_exclusive": END_DATE,
            "n_stations": len(data),
            "features": ",".join(FEATURES),
            "data_version_sha256": data_version,
        }
    )
    mlflow.log_metrics(
        {
            "silhouette_score": float(silhouette),
            "davies_bouldin_index": float(db_index),
            "inertia": inertia,
            "training_time_seconds": float(training_seconds),
            "n_clusters_observed": len(unique_labels),
        }
    )

    # Registrar el conjunto de características utilizado en el experimento
    dataset = mlflow.data.from_pandas(
        data[FEATURES].copy(),
        source="data/processed/viajes_*.parquet",
        name="p3_station_features",
    )

    mlflow.log_input(dataset, context="training")

    input_example = data[FEATURES].head(min(5, len(data))).copy()
    signature = infer_signature(input_example, model.predict(input_example))
    mlflow.sklearn.log_model(
        sk_model=model,
        name="modelo_kmeans",
        signature=signature,
        input_example=input_example,
    )

    log_data_manifest(file_hashes, data_version)
    log_candidate_config(config, stage, run_name)

    plot_data = data.copy()
    plot_data["cluster"] = labels
    log_cluster_plot(
        plot_data,
        run_name,
        f"P3: Segmentación de estaciones (K={n_clusters}, {stage})",
    )

    print(
        f"{run_name}: K={n_clusters} | silhouette={silhouette:.4f} | "
        f"Davies-Bouldin={db_index:.4f} | inertia={inertia:.3f}"
    )

    return {
        "run_name": run_name,
        "n_clusters": n_clusters,
        "silhouette_score": float(silhouette),
        "davies_bouldin_index": float(db_index),
        "inertia": inertia,
        "training_time_seconds": float(training_seconds),
    }


# ---------------------------------------------------------------------
# 7. Ejecutar baseline y búsqueda anidada
# ---------------------------------------------------------------------

def run_experiment():
    trips, file_hashes = load_trips()
    data = build_features(trips)
    X = data[FEATURES].copy()

    tracking_uri = configure_mlflow()
    git_commit = get_git_commit()
    data_version = get_data_version(file_hashes)

    # Baseline independiente, con el K configurado originalmente.
    with mlflow.start_run(run_name="P3_baseline_kmeans"):
        baseline_config = {
            "n_clusters": BASELINE_K,
            "n_init": 10,
            "max_iter": 300,
        }
        baseline_result = evaluate_candidate(
            X, data, baseline_config, "p3_baseline",
            "baseline", file_hashes, data_version, git_commit
        )

    # Run padre para agrupar la búsqueda; cada candidato es un child run.
    candidate_results = []
    with mlflow.start_run(run_name="P3_hyperparameter_search") as parent_run:
        mlflow.set_tags(
            {
                "team": EXPERIMENT_NAME,
                "question": "P3",
                "model_family": "clustering",
                "stage_of_work": "search",
                "git_commit": git_commit,
                "data_version": data_version,
            }
        )
        mlflow.log_params(
            {
                "search_type": "manual_grid",
                "candidate_count": len(CANDIDATE_CONFIGS),
                "baseline_n_clusters": BASELINE_K,
                "features": ",".join(FEATURES),
            }
        )

        for index, config in enumerate(CANDIDATE_CONFIGS, start=1):
            run_name = f"P3_candidate_k{config['n_clusters']}"
            with mlflow.start_run(run_name=run_name, nested=True):
                result = evaluate_candidate(
                    X, data, config, run_name, "candidate",
                    file_hashes, data_version, git_commit
                )
                candidate_results.append(result)

        # Selección orientativa: mayor Silhouette; desempate con menor
        # Davies-Bouldin. Inertia se reporta, pero no se compara sola entre K.
        best = sorted(
            candidate_results,
            key=lambda item: (
                -item["silhouette_score"],
                item["davies_bouldin_index"],
            ),
        )[0]

        mlflow.log_metrics(
            {
                "best_candidate_silhouette": best["silhouette_score"],
                "best_candidate_davies_bouldin": best["davies_bouldin_index"],
                "baseline_silhouette": baseline_result["silhouette_score"],
                "baseline_davies_bouldin": baseline_result["davies_bouldin_index"],
            }
        )
        mlflow.set_tag("selected_candidate", best["run_name"])
        mlflow.set_tag("selection_rule", "max_silhouette_then_min_davies_bouldin")

        comparison = {
            "baseline": baseline_result,
            "candidates": candidate_results,
            "selected_candidate": best,
            "selection_rule": (
                "Mayor Silhouette; en caso de empate, menor Davies-Bouldin. "
                "La selección debe revisarse también por interpretabilidad "
                "de los perfiles y utilidad para el negocio."
            ),
        }
        comparison_path = CONFIG_DIR / "p3_comparison_results.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(comparison_path), artifact_path="config")

    print("\n===== RESUMEN EXPERIMENTACIÓN P3 =====")
    print(f"Viajes analizados: {len(trips):,}")
    print(f"Estaciones analizadas: {len(data):,}")
    print(
        f"Baseline K={baseline_result['n_clusters']}: "
        f"Silhouette={baseline_result['silhouette_score']:.4f}, "
        f"Davies-Bouldin={baseline_result['davies_bouldin_index']:.4f}"
    )
    print(f"Candidato seleccionado por métricas: {best['run_name']}")
    print(f"MLflow URI: {tracking_uri}")
    print(
        "Nota: revisar también los perfiles de estaciones y su utilidad "
        "operativa antes de declarar un modelo final."
    )


if __name__ == "__main__":
    run_experiment()
