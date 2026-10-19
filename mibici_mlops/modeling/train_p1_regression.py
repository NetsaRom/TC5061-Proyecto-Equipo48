
from pathlib import Path
import hashlib
import os
import re
import unicodedata

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
    "mibici-equipo48"
)

MODEL_NAME = "RandomForestRegressor"
N_ESTIMATORS = 100
MAX_DEPTH = 15

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
# 3. CARGA DE ARCHIVOS CSV
# ============================================================

def load_csv(path):
    """Lee CSV intentando codificaciones habituales."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"No se pudo leer el archivo: {path}")


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
            [
                "Inicio_del_viaje",
                "Fecha_inicio",
                "fecha_hora_inicio",
                "start_time",
                "started_at",
            ],
        )

        origin_col = find_column(
            df,
            [
                "Origen_Id",
                "Estacion_de_Origen",
                "Estacion_origen",
                "id_estacion_origen",
                "start_station_id",
            ],
        )

        # Algunos datasets pueden usar fecha y hora separadas.
        if date_col is None:
            day_col = find_column(
                df, ["Fecha", "Fecha_del_viaje", "date"]
            )
            time_col = find_column(
                df, ["Hora_inicio", "Hora", "start_hour"]
            )

            if day_col is not None and time_col is not None:
                df["_fecha_inicio"] = (
                    df[day_col].astype(str).str.strip()
                    + " "
                    + df[time_col].astype(str).str.strip()
                )
                date_col = "_fecha_inicio"

        if date_col is None or origin_col is None:
            raise ValueError(
                f"No se detectaron las columnas necesarias en "
                f"{file.name}.\nColumnas disponibles: "
                f"{list(df.columns)}"
            )

        part = pd.DataFrame({
            "fecha_inicio": pd.to_datetime(
                df[date_col], errors="coerce", dayfirst=True
            ),
            "estacion": df[origin_col].astype("string").str.strip(),
        })

        part = part.dropna(
            subset=["fecha_inicio", "estacion"]
        )

        part = part[
            part["estacion"].ne("")
            & part["estacion"].ne("nan")
        ]

        frames.append(part)

    trips = pd.concat(frames, ignore_index=True)

    # Fechas de inicio incluidas; END_DATE es exclusiva.
    trips = trips[
        (trips["fecha_inicio"] >= pd.Timestamp(START_DATE))
        & (trips["fecha_inicio"] < pd.Timestamp(END_DATE))
    ].copy()

    if trips.empty:
        raise ValueError(
            "No hay viajes en el rango de fechas configurado. "
            "Revisa START_DATE y END_DATE."
        )

    return trips, hashes


# ============================================================
# 4. AGREGACIÓN Y FEATURE ENGINEERING
# ============================================================


def build_features(trips):
    """Genera variables por estación y hora para predecir
    los viajes de la siguiente hora.
    """

    trips = trips.copy()
    trips["hora"] = trips["fecha_inicio"].dt.floor("h")

    # 1. Contar viajes por estación y hora
    hourly = (
        trips.groupby(["estacion", "hora"])
        .size()
        .rename("viajes")
        .reset_index()
    )

    if hourly.empty:
        raise ValueError(
            "No hay viajes válidos para construir las features."
        )

    # 2. Construir una serie horaria por estación
    stations = hourly["estacion"].unique()
    series_list = []

    for station in stations:
        station_data = (
            hourly.loc[hourly["estacion"] == station]
            .set_index("hora")["viajes"]
            .sort_index()
        )

        full_index = pd.date_range(
            start=station_data.index.min(),
            end=station_data.index.max(),
            freq="h"
        )

        station_data = station_data.reindex(
            full_index,
            fill_value=0
        )

        part = station_data.rename("viajes").to_frame()
        part.index.name = "hora"
        part["estacion"] = station

        # Agregar cada estación una sola vez
        series_list.append(part.reset_index())

    # 3. Unir y ordenar los datos
    data = pd.concat(series_list, ignore_index=True)
    data = data.sort_values(
        ["estacion", "hora"]
    ).reset_index(drop=True)

    # 4. Variables de calendario
    data["hora_dia"] = data["hora"].dt.hour
    data["dia_semana"] = data["hora"].dt.dayofweek
    data["mes"] = data["hora"].dt.month
    data["fin_semana"] = (
        data["dia_semana"] >= 5
    ).astype(int)

    # 5. Variables históricas por estación
    grouped = data.groupby("estacion", sort=False)["viajes"]

    data["viajes_lag_1"] = grouped.shift(1)
    data["viajes_lag_24"] = grouped.shift(24)
    data["viajes_lag_168"] = grouped.shift(168)

    data["promedio_24h"] = grouped.transform(
        lambda s: s.shift(1).rolling(
            window=24,
            min_periods=1
        ).mean()
    )

    # 6. Objetivo: viajes de la siguiente hora
    data["target_viajes_siguiente_hora"] = (
        grouped.shift(-1)
    )

    # 7. Diagnóstico
    print("\n--- DIAGNÓSTICO DE FEATURES ---")
    print(f"Filas creadas: {len(data):,}")
    print(f"Estaciones: {data['estacion'].nunique():,}")
    print(f"Primera hora: {data['hora'].min()}")
    print(f"Última hora: {data['hora'].max()}")

    columnas = [
        "viajes_lag_1",
        "viajes_lag_24",
        "viajes_lag_168",
        "promedio_24h",
        "target_viajes_siguiente_hora"
    ]

    print("\nValores nulos por variable:")
    print(data[columnas].isna().sum())

    # 8. Eliminar filas sin historial inmediato
    # o sin valor objetivo
    data = data.dropna(
        subset=[
            "viajes_lag_1",
            "target_viajes_siguiente_hora"
        ]
    ).copy()

    if data.empty:
        raise ValueError(
            "No quedaron filas para entrenar. "
            "Revisa las fechas y los archivos CSV."
        )

    print(f"\nFilas finales para modelado: {len(data):,}")

    return data

# ============================================================
# 5. BASELINE Y MODELO
# ============================================================

def run_experiment():
    trips, file_hashes = load_trips()
    data = build_features(trips)

    features = [
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

    target = "target_viajes_siguiente_hora"

    # División cronológica por hora, no aleatoria.
    timestamps = np.sort(data["hora"].dropna().unique())
    if len(timestamps) < 2:
        raise ValueError(
            f"Solo hay {len(timestamps)} hora(s) disponibles después de "
            "crear las features. Se necesitan al menos dos para separar "
            "entrenamiento y prueba. Descarga más datos horarios."
        )

    cutoff_idx = int(len(timestamps) * 0.8)
    cutoff_idx = min(max(cutoff_idx, 1), len(timestamps) - 1)
    cutoff = timestamps[cutoff_idx]

    train = data[data["hora"] < cutoff].copy()
    test = data[data["hora"] >= cutoff].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"No se pudo realizar la división temporal. Horas disponibles: "
            f"{len(timestamps)}; corte: {pd.Timestamp(cutoff)}."
        )

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    # Baseline: repetir la demanda de la hora anterior.
    baseline_predictions = test["viajes_lag_1"].to_numpy()

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )
    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    categorical = ["estacion"]
    numeric = [col for col in features if col not in categorical]

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
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])

    # ========================================================
    # 6. REGISTRO EN MLFLOW
    # ========================================================

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    git_commit = os.getenv("GIT_COMMIT", "unknown")

    with mlflow.start_run(run_name="P1_random_forest_regression"):
        mlflow.set_tags({
            "team": EXPERIMENT_NAME,
            "question": "P1",
            "git_commit": git_commit,
            "data_version": "sha256_manifest",
        })

        mlflow.log_params({
            "start_date": START_DATE,
            "end_date_exclusive": END_DATE,
            "cutoff": str(pd.Timestamp(cutoff)),
            "n_train": len(train),
            "n_test": len(test),
            "n_stations": int(data["estacion"].nunique()),
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "random_state": RANDOM_STATE,
            "target": target,
            "features": ",".join(features),
            "data_version_sha256": hashlib.sha256(
                "".join(f"{name}:{digest}" for name, digest in sorted(file_hashes.items())).encode("utf-8")
            ).hexdigest(),
        })

        mlflow.log_metrics({
            "baseline_test_mae": baseline_mae,
            "baseline_test_rmse": baseline_rmse,
        })

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))


        # Evaluación por estación
        estacion_ejemplo = test["estacion"].iloc[0]

        station_mask = (
            test["estacion"].astype(str).to_numpy()
            == str(estacion_ejemplo)
        )

        y_station_real = y_test.to_numpy()[station_mask]
        y_station_pred = predictions[station_mask]

        print("\n--- EVALUACIÓN POR ESTACIÓN ---")
        print("Estación:", estacion_ejemplo)
        print("Observaciones:", len(y_station_real))
        print(
            "MAE estación:",
            f"{mean_absolute_error(y_station_real, y_station_pred):.3f}"
        )
        print(
            "RMSE estación:",
            f"{np.sqrt(mean_squared_error(y_station_real, y_station_pred)):.3f}"
        )

        mlflow.log_metrics({
            "test_mae": mae,
            "test_rmse": rmse,
        })

        # Gráfica de predicciones frente a valores reales
        sample = test[["hora", "estacion"]].copy()
        sample["real"] = y_test.to_numpy()
        sample["predicho"] = predictions

        # Seleccionar una sola estación para visualizar una serie temporal
        estacion_ejemplo = sample["estacion"].iloc[0]

        sample = sample[
            sample["estacion"] == estacion_ejemplo
        ].sort_values("hora").head(300)

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(
            sample["hora"],
            sample["real"],
            label="Viajes reales",
            marker=".",
            markersize=4
        )

        ax.plot(
            sample["hora"],
            sample["predicho"],
            label="Viajes predichos",
            marker=".",
            markersize=4
        )

        ax.set_title(
            f"P1: viajes reales vs. predichos - Estación {estacion_ejemplo}"
        )
        ax.set_xlabel("Fecha y hora")
        ax.set_ylabel("Número de viajes")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        fig.tight_layout()

        figure_path = REPORTS_DIR / "p1_regression.png"
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

        mlflow.log_artifact(str(figure_path))

        print("\n===== RESULTADOS P1 =====")
        print(f"Viajes cargados: {len(trips):,}")
        print(f"Estaciones: {data['estacion'].nunique():,}")
        print(f"Entrenamiento: {len(train):,}")
        print(f"Prueba: {len(test):,}")
        print(f"Baseline MAE: {baseline_mae:.3f}")
        print(f"Baseline RMSE: {baseline_rmse:.3f}")
        print(f"Random Forest MAE: {mae:.3f}")
        print(f"Random Forest RMSE: {rmse:.3f}")
        print(f"Gráfica: {figure_path}")
        print(f"MLflow URI: {tracking_uri}")


if __name__ == "__main__":
    run_experiment()