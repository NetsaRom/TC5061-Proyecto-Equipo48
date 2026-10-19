from pathlib import Path
import hashlib
import os
import re
import unicodedata

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports" / "figures"

START_DATE = "2026-07-01"
END_DATE = "2026-10-01"
RANDOM_STATE = 42

EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "mibici-equipoNN"
)

MODEL_NAME = "KMeansClustering"
N_CLUSTERS = 4  # Número de grupos adaptado a la lógica del negocio

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. NORMALIZACIÓN DE COLUMNAS
# ============================================================

def normalize_column(name):
    """Normaliza nombres para facilitar la detección de columnas."""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def find_column(df, candidates):
    """Encuentra una columna usando nombres alternativos."""
    normalized = {
        normalize_column(col): col for col in df.columns
    }

    for candidate in candidates:
        key = normalize_column(candidate)
        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# 3. CARGA Y FILTRADO DE DATOS
# ============================================================

def load_trips():
    """Carga y normaliza los archivos mensuales de viajes."""
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
            ["Inicio_del_viaje", "Fecha_inicio", "fecha_hora_inicio", "start_time", "started_at"]
        )

        origin_col = find_column(
            df,
            ["Origen_Id", "Estacion_de_Origen", "Estacion_origen", "id_estacion_origen", "start_station_id"]
        )
        
        duration_col = find_column(
            df,
            ["duracion_min", "duration_minutes", "trip_duration_min"]
        )
        
        anomaly_col = find_column(
            df,
            ["duracion_atipica", "is_anomaly", "anomaly"]
        )

        if date_col is None or origin_col is None or duration_col is None:
            raise ValueError(
                f"No se detectaron las columnas necesarias en {file.name}.\n"
                f"Columnas disponibles: {list(df.columns)}"
            )

        # Filtrado de datos atípicos para no distorsionar centroides
        if anomaly_col is not None:
            df = df[df[anomaly_col] == False]

        part = pd.DataFrame({
            "fecha_inicio": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True),
            "estacion": df[origin_col].astype("string").str.strip(),
            "duracion_min": df[duration_col].astype(float)
        })

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


# ============================================================
# 4. AGREGACIÓN E INGENIERÍA DE CARACTERÍSTICAS
# ============================================================

def build_features(trips):
    """Genera variables consolidadas por estación para clustering."""
    trips = trips.copy()
    trips["hora_dia"] = trips["fecha_inicio"].dt.hour

    logger_msg = "Calculando métricas de comportamiento agregado por estación..."
    print(f"\n--- {logger_msg} ---")

    # 1. Agrupar por estación base
    grouped = trips.groupby("estacion")

    # 2. Construir métricas clave
    total_viajes = grouped.size().rename("viajes_salientes")
    promedio_duracion = grouped["duracion_min"].mean().rename("promedio_duracion_salida")
    
    # Viajes en hora pico matutina (6 AM - 9 AM)
    viajes_matutinos = trips[(trips["hora_dia"] >= 6) & (trips["hora_dia"] <= 9)].groupby("estacion").size()
    pct_matutino = (viajes_matutinos / total_viajes).fillna(0).rename("pct_viajes_matutinos")

    # 3. Consolidar dataset final de estaciones
    data = pd.concat([total_viajes, promedio_duracion, pct_matutino], axis=1).reset_index()
    data = data.dropna().reset_index(drop=True)

    print(f"Estaciones identificadas para segmentación: {len(data):,}")
    return data


# ============================================================
# 5. EXPERIMENTO DE CLUSTERING NO SUPERVISADO
# ============================================================

def run_experiment():
    trips, file_hashes = load_trips()
    data = build_features(trips)

    features = ["viajes_salientes", "promedio_duracion_salida", "pct_viajes_matutinos"]
    X = data[features].to_numpy()

    # Escalamiento estándar obligatorio para distancias euclidianas (K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Configuración del Tracking URI nativo del repositorio
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    git_commit = os.getenv("GIT_COMMIT", "unknown")

    with mlflow.start_run(run_name="P3_kmeans_clustering"):
        mlflow.set_tags({
            "team": EXPERIMENT_NAME,
            "question": "P3",
            "git_commit": git_commit,
            "data_version": "sha256_manifest",
        })

        mlflow.log_params({
            "start_date": START_DATE,
            "end_date_exclusive": END_DATE,
            "n_stations": len(data),
            "algoritmo": MODEL_NAME,
            "n_clusters": N_CLUSTERS,
            "random_state": RANDOM_STATE,
            "features_utilizadas": ",".join(features),
            "data_version_sha256": hashlib.sha256(
                "".join(f"{name}:{digest}" for name, digest in sorted(file_hashes.items())).encode("utf-8")
            ).hexdigest(),
        })

        # Entrenar K-Means
        model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = model.fit_predict(X_scaled)
        data["cluster"] = cluster_labels

        # Evaluación Geométrica Intrínseca
        silhouette = silhouette_score(X_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)

        mlflow.log_metrics({
            "silhouette_score": silhouette,
            "davies_bouldin_index": davies_bouldin,
            "inertia": model.inertia_,
        })

        # Registrar el modelo entrenado
        mlflow.sklearn.log_model(model, "modelo_kmeans")

        # Generación de la gráfica de visualización de perfiles
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            data["viajes_salientes"],
            data["promedio_duracion_salida"],
            c=data["cluster"],
            cmap="viridis",
            alpha=0.6,
            edgecolors="w",
            s=50
        )
        
        ax.set_title(f"P3: Segmentación de Estaciones (K={N_CLUSTERS})")
        ax.set_xlabel("Volumen de Viajes Salientes")
        ax.set_ylabel("Duración Promedio de Salida (min)")
        
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        figure_path = REPORTS_DIR / "p3_clustering.png"
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

        mlflow.log_artifact(str(figure_path))

        print("\n===== RESULTADOS P3 =====")
        print(f"Viajes totales leídos: {len(trips):,}")
        print(f"Estaciones procesadas: {len(data):,}")
        print(f"K-Means Inertia: {model.inertia_:.3f}")
        print(f"Silhouette Score (Calidad): {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"Gráfica guardada en: {figure_path}")
        print(f"MLflow URI: {tracking_uri}\n")


if __name__ == "__main__":
    run_experiment()
