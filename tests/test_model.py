from types import SimpleNamespace

import numpy as np
import pytest

from mibici_mlops.api import model as model_module
from mibici_mlops.api.model import ModelService

VALID_FEATURES = {
    "estacion": "TEST",
    "hora_dia": 10,
    "dia_semana": 2,
    "mes": 9,
    "fin_semana": 0,
    "viajes_lag_1": 3.0,
    "viajes_lag_24": 4.0,
    "viajes_lag_168": 5.0,
    "promedio_24h": 4.0,
}


def test_load_with_tracking_uri(monkeypatch):
    service = ModelService()
    fake_model = object()
    tracking_uris = []
    loaded_uris = []

    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///test-mlflow.db",
    )
    monkeypatch.setattr(
        model_module.mlflow,
        "set_tracking_uri",
        lambda uri: tracking_uris.append(uri),
    )
    monkeypatch.setattr(
        model_module.mlflow,
        "pyfunc",
        SimpleNamespace(load_model=lambda uri: loaded_uris.append(uri) or fake_model),
    )

    result = service.load()

    assert result is fake_model
    assert service.is_loaded is True
    assert tracking_uris == ["sqlite:///test-mlflow.db"]
    assert loaded_uris == [model_module.MODEL_URI]


def test_load_with_default_tracking_uri(monkeypatch):
    service = ModelService()
    tracking_uris = []

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        model_module.mlflow,
        "set_tracking_uri",
        lambda uri: tracking_uris.append(uri),
    )
    monkeypatch.setattr(
        model_module.mlflow,
        "pyfunc",
        SimpleNamespace(load_model=lambda uri: object()),
    )

    service.load()

    assert len(tracking_uris) == 1
    assert tracking_uris[0].endswith("/mlflow.db")


def test_is_loaded_initially_false():
    service = ModelService()

    assert service.is_loaded is False


def test_predict_without_loaded_model():
    service = ModelService()

    with pytest.raises(
        RuntimeError,
        match="El modelo no está cargado.",
    ):
        service.predict(VALID_FEATURES)


def test_predict_success_and_input_dtypes():
    service = ModelService()
    captured_data = {}

    class FakeModel:
        def predict(self, data):
            captured_data["data"] = data
            return np.array([4.5])

    service.model = FakeModel()

    result = service.predict(VALID_FEATURES)
    data = captured_data["data"]

    assert result == 4.5
    assert isinstance(result, float)
    assert data.shape == (1, 9)

    assert isinstance(data["estacion"].iloc[0], str)
    assert data["estacion"].iloc[0] == "TEST"
    assert data["hora_dia"].dtype == np.int32
    assert data["dia_semana"].dtype == np.int32
    assert data["mes"].dtype == np.int32
    assert data["fin_semana"].dtype == np.int64

    assert data["viajes_lag_1"].dtype == np.float64
    assert data["viajes_lag_24"].dtype == np.float64
    assert data["viajes_lag_168"].dtype == np.float64
    assert data["promedio_24h"].dtype == np.float64
