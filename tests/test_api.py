from fastapi.testclient import TestClient
import pytest

from mibici_mlops.api import main

VALID_PAYLOAD = {
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


@pytest.fixture
def client(monkeypatch):
    """Cliente de pruebas con un modelo simulado."""
    monkeypatch.setattr(
        main.model_service,
        "load",
        lambda: setattr(main.model_service, "model", object()),
    )

    with TestClient(main.app) as test_client:
        yield test_client


def test_health_success(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_success(client, monkeypatch):
    monkeypatch.setattr(
        main.model_service,
        "predict",
        lambda features: 4.5,
    )

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prediction"] == 4.5
    assert response.json()["unit"] == "viajes"


def test_predict_invalid_hour(client):
    payload = {**VALID_PAYLOAD, "hora_dia": 25}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_missing_field(client):
    payload = VALID_PAYLOAD.copy()
    payload.pop("estacion")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_extra_field(client):
    payload = {**VALID_PAYLOAD, "campo_extra": "incorrecto"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_negative_values(client):
    payload = {**VALID_PAYLOAD, "viajes_lag_1": -1.0}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_model_unavailable(client, monkeypatch):
    monkeypatch.setattr(main.model_service, "model", None)

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 503


def test_health_model_unavailable(client, monkeypatch):
    monkeypatch.setattr(main.model_service, "model", None)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"]["model_loaded"] is False


def test_predict_internal_error(client, monkeypatch):
    def raise_error(features):
        raise RuntimeError("Error simulado")

    monkeypatch.setattr(main.model_service, "predict", raise_error)

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 500
    assert response.json()["detail"] == ("Error interno al ejecutar la predicción.")


def test_lifespan_model_load_failure(monkeypatch):
    def raise_error():
        raise RuntimeError("MLflow no disponible")

    monkeypatch.setattr(
        main.model_service,
        "load",
        raise_error,
    )

    with TestClient(main.app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"]["model_loaded"] is False
    assert response.json()["detail"]["status"] == "model_unavailable"
