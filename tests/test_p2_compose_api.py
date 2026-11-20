from fastapi.testclient import TestClient
import pytest
import requests

from api import main


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload or {}
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


VALID_PAYLOAD = {
    "promedio_duracion_salida": 9.25,
    "pct_viajes_matutinos": 0.24,
    "viajes_entrantes": 1452,
}


class FakeModel:
    def __init__(self, prediction=1):
        self.prediction = prediction
        self.received = None

    def predict(self, dataframe):
        self.received = dataframe
        return [self.prediction]


@pytest.fixture
def client(monkeypatch):
    """Modo remoto (Docker Compose): la API delega en MLflow Model Serving."""
    monkeypatch.setenv("MLFLOW_SERVING_URL", "http://model-serving:5001")
    monkeypatch.setattr(main.requests, "get", lambda *args, **kwargs: FakeResponse())
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def local_client(monkeypatch):
    """Modo local (docker run): sin MLFLOW_SERVING_URL se carga el modelo en la API."""
    monkeypatch.delenv("MLFLOW_SERVING_URL", raising=False)
    fake_model = FakeModel(prediction=1)
    monkeypatch.setattr(main, "load_local_model", lambda: fake_model)

    def no_http(*args, **kwargs):
        raise AssertionError("El modo local no debe hacer llamadas HTTP")

    monkeypatch.setattr(main.requests, "get", no_http)
    monkeypatch.setattr(main.requests, "post", no_http)
    with TestClient(main.app) as test_client:
        yield test_client, fake_model


def test_local_mode_health_reports_local(local_client):
    test_client, _ = local_client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["modo"] == "local"


def test_local_mode_predicts_with_in_process_model(local_client):
    test_client, fake_model = local_client

    response = test_client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["prediccion"] == 1
    assert list(fake_model.received.columns) == main.FEATURES


def test_remote_mode_health_reports_remote(client):
    response = client.get("/health")

    assert response.json()["modo"] == "remoto"


def test_health_checks_mlflow_model_serving(client, monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return FakeResponse()

    monkeypatch.setattr(main.requests, "get", fake_get)

    response = client.get("/health")

    assert response.status_code == 200
    assert calls[0][0].endswith("/ping")


def test_predict_forwards_dataframe_split_to_mlflow(client, monkeypatch):
    request_data = {}

    def fake_post(url, json, timeout):
        request_data["url"] = url
        request_data["json"] = json
        return FakeResponse({"predictions": [1]})

    monkeypatch.setattr(main.requests, "post", fake_post)

    response = client.post(
        "/predict",
        json={
            "promedio_duracion_salida": 9.25,
            "pct_viajes_matutinos": 0.24,
            "viajes_entrantes": 1452,
        },
    )

    assert response.status_code == 200
    assert response.json()["prediccion"] == 1
    assert request_data["url"].endswith("/invocations")
    assert request_data["json"]["dataframe_split"]["columns"] == main.FEATURES


def test_predict_returns_503_when_model_serving_is_unavailable(client, monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("model-serving unavailable")

    monkeypatch.setattr(main.requests, "post", unavailable)

    response = client.post(
        "/predict",
        json={
            "promedio_duracion_salida": 9.25,
            "pct_viajes_matutinos": 0.24,
            "viajes_entrantes": 1452,
        },
    )

    assert response.status_code == 503
