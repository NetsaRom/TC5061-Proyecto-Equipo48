import json

import numpy as np
import pandas as pd
import pytest

from mibici_mlops.monitoring.p2_drift import (
    FEATURES,
    REQUIRED_COLUMNS,
    load_dataset,
    run_monitoring,
    validate_dataset,
)


def make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "estacion": pd.Series([1, 2, 3, 4, 5], dtype="int64"),
            "promedio_duracion_salida": [8.0, 9.0, 10.0, 11.0, 12.0],
            "pct_viajes_matutinos": [20.0, 25.0, 30.0, 35.0, 40.0],
            "viajes_entrantes": pd.Series([30, 40, 50, 60, 70], dtype="uint32"),
        }
    )


def write_dataset(path, dataset=None):
    (dataset if dataset is not None else make_dataset()).to_parquet(path, index=False)


def test_load_dataset_reads_parquet(tmp_path):
    path = tmp_path / "reference.parquet"
    write_dataset(path)

    loaded = load_dataset(path)

    assert loaded.columns.tolist() == REQUIRED_COLUMNS
    assert loaded.shape == (5, 4)


def test_load_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No se encontró"):
        load_dataset(tmp_path / "missing.parquet")


def test_validate_dataset_rejects_missing_or_unexpected_columns():
    dataset = make_dataset().drop(columns="estacion")

    with pytest.raises(ValueError, match="contrato de columnas"):
        validate_dataset(dataset, "Reference")


def test_validate_dataset_rejects_null_values():
    dataset = make_dataset()
    dataset.loc[0, FEATURES[0]] = np.nan

    with pytest.raises(ValueError, match="valores nulos"):
        validate_dataset(dataset, "Reference")


def test_validate_dataset_rejects_non_numeric_features():
    dataset = make_dataset()
    dataset[FEATURES[1]] = "invalid"

    with pytest.raises(TypeError, match="no numéricas"):
        validate_dataset(dataset, "Reference")


def test_validate_dataset_rejects_infinite_values():
    dataset = make_dataset()
    dataset[FEATURES[2]] = dataset[FEATURES[2]].astype("float64")
    dataset.loc[0, FEATURES[2]] = np.inf

    with pytest.raises(ValueError, match="infinitos"):
        validate_dataset(dataset, "Reference")


def test_run_monitoring_generates_report_and_excludes_identifier(tmp_path):
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    write_dataset(reference_path)
    write_dataset(
        current_path,
        make_dataset().assign(viajes_entrantes=lambda df: df.viajes_entrantes + 2),
    )

    artifacts = run_monitoring(
        reference_path=reference_path,
        current_path=current_path,
        html_report_path=tmp_path / "drift.html",
        evidently_snapshot_path=tmp_path / "snapshot.json",
        quality_summary_path=tmp_path / "quality.json",
    )

    assert artifacts.html_report.is_file()
    assert artifacts.evidently_snapshot.is_file()
    summary = json.loads(artifacts.quality_summary.read_text(encoding="utf-8"))
    assert summary["evidently"]["features_analyzed"] == FEATURES
    assert summary["evidently"]["identifier_excluded"] == "estacion"
    assert set(summary["evidently"]["drift"]["per_feature"]) == set(FEATURES)
