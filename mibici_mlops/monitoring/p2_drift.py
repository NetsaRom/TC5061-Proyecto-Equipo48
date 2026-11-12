"""Monitoreo de calidad y data drift para las variables predictoras de P2.

El módulo compara los datasets agregados por estación preparados para P2. La
columna ``estacion`` se usa únicamente para revisar cobertura y duplicados; no
se entrega a Evidently ni participa en los tests estadísticos.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import numpy as np
import pandas as pd
import typer

from mibici_mlops.config import INTERIM_DATA_DIR, REPORTS_DIR

app = typer.Typer(no_args_is_help=True)

FEATURES = [
    "promedio_duracion_salida",
    "pct_viajes_matutinos",
    "viajes_entrantes",
]
IDENTIFIER = "estacion"
REQUIRED_COLUMNS = [IDENTIFIER, *FEATURES]
REFERENCE_PERIOD_DAYS = 92
CURRENT_PERIOD_DAYS = 31
KS_THRESHOLD = 0.05


@dataclass(frozen=True)
class MonitoringArtifacts:
    """Rutas de los artefactos generados por una ejecución de monitoreo."""

    html_report: Path
    evidently_snapshot: Path
    quality_summary: Path


def load_dataset(path: Path) -> pd.DataFrame:
    """Carga un Parquet de drift y verifica que exista."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")
    return pd.read_parquet(path)


def validate_dataset(dataset: pd.DataFrame, dataset_name: str) -> None:
    """Valida contrato, tipos y valores inválidos de un dataset de P2."""
    actual_columns = set(dataset.columns)
    expected_columns = set(REQUIRED_COLUMNS)
    if actual_columns != expected_columns:
        missing = sorted(expected_columns - actual_columns)
        unexpected = sorted(actual_columns - expected_columns)
        raise ValueError(
            f"{dataset_name} no cumple el contrato de columnas. "
            f"Faltantes: {missing}; inesperadas: {unexpected}."
        )

    if not pd.api.types.is_integer_dtype(dataset[IDENTIFIER]):
        raise TypeError(f"{dataset_name}.{IDENTIFIER} debe ser un identificador entero.")

    non_numeric = [
        feature for feature in FEATURES if not pd.api.types.is_numeric_dtype(dataset[feature])
    ]
    if non_numeric:
        raise TypeError(f"{dataset_name} contiene características no numéricas: {non_numeric}.")

    null_counts = dataset[REQUIRED_COLUMNS].isna().sum()
    if null_counts.any():
        details = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"{dataset_name} contiene valores nulos: {details}.")

    values = dataset[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{dataset_name} contiene valores infinitos.")


def validate_pair(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Comprueba que ambos datasets cumplan el mismo contrato de P2."""
    validate_dataset(reference, "Reference")
    validate_dataset(current, "Current")


def _descriptive_stats(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "max": float(series.max()),
    }


def build_quality_summary(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    """Construye un resumen trazable de calidad y cobertura de ambos conjuntos."""
    reference_stations = set(reference[IDENTIFIER])
    current_stations = set(current[IDENTIFIER])

    summary: dict[str, Any] = {
        "datasets": {},
        "station_coverage": {
            "shared_count": len(reference_stations & current_stations),
            "only_reference": sorted(reference_stations - current_stations),
            "only_current": sorted(current_stations - reference_stations),
        },
        "features": {},
        "methodological_context": {
            "reference_period_days": REFERENCE_PERIOD_DAYS,
            "current_period_days": CURRENT_PERIOD_DAYS,
            "viajes_entrantes_note": (
                "Los conteos por día son contexto descriptivo y no se usan como "
                "entrada de Evidently ni modifican los datasets originales."
            ),
        },
    }

    for name, dataset in (("reference", reference), ("current", current)):
        summary["datasets"][name] = {
            "rows": len(dataset),
            "columns": list(dataset.columns),
            "dtypes": {column: str(dtype) for column, dtype in dataset.dtypes.items()},
            "null_count": int(dataset.isna().sum().sum()),
            "infinite_count": int(np.isinf(dataset[FEATURES].to_numpy(dtype=float)).sum()),
            "duplicate_row_count": int(dataset.duplicated().sum()),
            "unique_station_count": int(dataset[IDENTIFIER].nunique()),
        }

    for feature in FEATURES:
        reference_stats = _descriptive_stats(reference[feature])
        current_stats = _descriptive_stats(current[feature])
        summary["features"][feature] = {
            "reference": reference_stats,
            "current": current_stats,
            "mean_difference_current_minus_reference": (
                current_stats["mean"] - reference_stats["mean"]
            ),
        }

    summary["features"]["viajes_entrantes"]["daily_context"] = {
        "reference_mean_per_day": (
            summary["features"]["viajes_entrantes"]["reference"]["mean"] / REFERENCE_PERIOD_DAYS
        ),
        "current_mean_per_day": (
            summary["features"]["viajes_entrantes"]["current"]["mean"] / CURRENT_PERIOD_DAYS
        ),
    }
    return summary


def _feature_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia numérica para Evidently sin el identificador de estación."""
    return dataset.loc[:, FEATURES].astype("float64")


def _extract_drift_results(snapshot: Any) -> dict[str, Any]:
    """Extrae resultados KS del Snapshot de Evidently 0.7 para documentarlos."""
    metrics = snapshot.dict()["metrics"]
    drift_results: dict[str, Any] = {"per_feature": {}}

    for metric in metrics:
        config = metric.get("config", {})
        if config.get("type", "").endswith("DriftedColumnsCount"):
            drift_results["dataset"] = metric["value"]
        elif config.get("type", "").endswith("ValueDrift"):
            p_value = float(metric["value"])
            drift_results["per_feature"][config["column"]] = {
                "method": config["method"],
                "threshold": float(config["threshold"]),
                "p_value": p_value,
                "drift_detected": p_value < float(config["threshold"]),
            }
    return drift_results


def run_monitoring(
    reference_path: Path,
    current_path: Path,
    html_report_path: Path,
    evidently_snapshot_path: Path,
    quality_summary_path: Path,
) -> MonitoringArtifacts:
    """Ejecuta las validaciones, Evidently y guarda artefactos reproducibles."""
    reference = load_dataset(reference_path)
    current = load_dataset(current_path)
    validate_pair(reference, current)

    report = Report(
        [
            DataDriftPreset(
                columns=FEATURES,
                num_method="ks",
                num_threshold=KS_THRESHOLD,
            ),
            DataSummaryPreset(columns=FEATURES),
        ]
    )
    snapshot = report.run(
        reference_data=_feature_frame(reference),
        current_data=_feature_frame(current),
    )

    for output_path in (html_report_path, evidently_snapshot_path, quality_summary_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot.save_html(str(html_report_path))
    snapshot.save_json(str(evidently_snapshot_path))

    quality_summary = build_quality_summary(reference, current)
    quality_summary["evidently"] = {
        "version": version("evidently"),
        "features_analyzed": FEATURES,
        "identifier_excluded": IDENTIFIER,
        "drift": _extract_drift_results(snapshot),
    }
    quality_summary_path.write_text(
        json.dumps(quality_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return MonitoringArtifacts(
        html_report=html_report_path,
        evidently_snapshot=evidently_snapshot_path,
        quality_summary=quality_summary_path,
    )


@app.callback()
def main() -> None:
    """Agrupa los comandos de monitoreo del modelo P2."""


@app.command()
def monitor(
    reference_path: Path = INTERIM_DATA_DIR / "p2_drift_reference.parquet",
    current_path: Path = INTERIM_DATA_DIR / "p2_drift_current.parquet",
    html_report_path: Path = REPORTS_DIR / "p2_evidently_drift_report.html",
    evidently_snapshot_path: Path = REPORTS_DIR / "p2_evidently_snapshot.json",
    quality_summary_path: Path = REPORTS_DIR / "p2_data_quality_summary.json",
) -> None:
    """Genera el reporte de drift y los resúmenes de calidad para P2."""
    artifacts = run_monitoring(
        reference_path=reference_path,
        current_path=current_path,
        html_report_path=html_report_path,
        evidently_snapshot_path=evidently_snapshot_path,
        quality_summary_path=quality_summary_path,
    )
    typer.echo(f"Reporte HTML: {artifacts.html_report}")
    typer.echo(f"Snapshot Evidently: {artifacts.evidently_snapshot}")
    typer.echo(f"Resumen de calidad: {artifacts.quality_summary}")


if __name__ == "__main__":
    app()
