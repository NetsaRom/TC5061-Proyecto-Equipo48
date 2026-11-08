"""
Pruebas de extremo a extremo para validar la integración:

Cliente -> FastAPI en Docker -> MLflow Model Serving -> modelo P2

Requisitos antes de ejecutar:
1. MLflow Tracking Server activo en el puerto 5000.
2. MLflow Model Serving activo en el puerto 5001.
3. Contenedor mibici-api-p2 activo en el puerto 8000.
"""

import os

import pytest
import requests


# Dirección pública de FastAPI.
# Puede modificarse con la variable de entorno API_BASE_URL.
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
).rstrip("/")

# Tiempo máximo de espera para cada solicitud HTTP.
TIMEOUT_SECONDS = 15


@pytest.fixture(scope="session")
def api_url() -> str:
    """
    Devuelve la URL base de FastAPI y verifica que no esté vacía.
    """

    assert API_BASE_URL, (
        "La variable API_BASE_URL no puede estar vacía."
    )

    return API_BASE_URL


@pytest.fixture(scope="session")
def valid_payload() -> dict:
    """
    Datos válidos para realizar una predicción.
    """

    return {
        "promedio_duracion_salida": 9.2547,
        "pct_viajes_matutinos": 0.2379,
        "viajes_entrantes": 1452,
    }


@pytest.fixture(scope="session")
def invalid_payload() -> dict:
    """
    Datos que incumplen las restricciones del contrato.
    """

    return {
        "promedio_duracion_salida": -10,
        "pct_viajes_matutinos": 1.5,
        "viajes_entrantes": -5,
    }


def test_health_fastapi_mlflow(api_url: str) -> None:
    """
    Verifica la comunicación entre FastAPI y MLflow Model Serving.

    Resultado esperado:
    - HTTP 200.
    - Estado general igual a "ok".
    - MLflow Model Serving disponible.
    - Modelo y alias correctos.
    """

    response = requests.get(
        f"{api_url}/health",
        timeout=TIMEOUT_SECONDS,
    )

    assert response.status_code == 200, (
        "El endpoint /health no respondió correctamente. "
        f"HTTP recibido: {response.status_code}. "
        f"Respuesta: {response.text}"
    )

    data = response.json()

    assert data["status"] == "ok"
    assert data["serving"] == "disponible"
    assert data["modelo"] == "mibici-equipo48-p2-riesgo"
    assert data["version"] == "1"


def test_predict_con_datos_validos(
    api_url: str,
    valid_payload: dict,
) -> None:
    """
    Verifica una predicción completa con datos válidos.

    La solicitud recorre:
    FastAPI -> MLflow Model Serving -> modelo P2.
    """

    response = requests.post(
        f"{api_url}/predict",
        json=valid_payload,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT_SECONDS,
    )

    assert response.status_code == 200, (
        "La predicción con datos válidos no fue exitosa. "
        f"HTTP recibido: {response.status_code}. "
        f"Respuesta: {response.text}"
    )

    data = response.json()

    assert data["modelo"] == "mibici-equipo48-p2-riesgo"
    assert data["version"] == "1"

    assert data["prediccion"] in (0, 1), (
        "La predicción debe ser 0 o 1. "
        f"Valor recibido: {data['prediccion']}"
    )

    expected_descriptions = {
        0: "Demanda no alta",
        1: "Alta demanda",
    }

    expected_description = expected_descriptions[
        data["prediccion"]
    ]

    assert data["descripcion"] == expected_description, (
        "La descripción no corresponde con la predicción. "
        f"Predicción: {data['prediccion']}. "
        f"Descripción recibida: {data['descripcion']}."
    )


def test_predict_rechaza_datos_invalidos(
    api_url: str,
    invalid_payload: dict,
) -> None:
    """
    Verifica que FastAPI rechace datos fuera del contrato.

    Los tres valores enviados son inválidos:
    - Duración promedio negativa.
    - Porcentaje mayor que 1.
    - Cantidad de viajes negativa.
    """

    response = requests.post(
        f"{api_url}/predict",
        json=invalid_payload,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT_SECONDS,
    )

    assert response.status_code == 422, (
        "FastAPI debería rechazar los datos inválidos con HTTP 422. "
        f"HTTP recibido: {response.status_code}. "
        f"Respuesta: {response.text}"
    )

    data = response.json()

    assert "detail" in data
    assert isinstance(data["detail"], list)
    assert len(data["detail"]) >= 3


def test_predict_rechaza_campo_faltante(
    api_url: str,
) -> None:
    """
    Verifica que FastAPI rechace una solicitud incompleta.
    """

    incomplete_payload = {
        "promedio_duracion_salida": 9.2547,
        "pct_viajes_matutinos": 0.2379,
        # Falta viajes_entrantes.
    }

    response = requests.post(
        f"{api_url}/predict",
        json=incomplete_payload,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT_SECONDS,
    )

    assert response.status_code == 422, (
        "FastAPI debería rechazar una solicitud incompleta "
        "con HTTP 422. "
        f"HTTP recibido: {response.status_code}. "
        f"Respuesta: {response.text}"
    )


def test_predict_rechaza_tipo_incorrecto(
    api_url: str,
) -> None:
    """
    Verifica que FastAPI rechace valores no numéricos.
    """

    incorrect_payload = {
        "promedio_duracion_salida": "dato-invalido",
        "pct_viajes_matutinos": "dato-invalido",
        "viajes_entrantes": "dato-invalido",
    }

    response = requests.post(
        f"{api_url}/predict",
        json=incorrect_payload,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT_SECONDS,
    )

    assert response.status_code == 422, (
        "FastAPI debería rechazar tipos incorrectos "
        "con HTTP 422. "
        f"HTTP recibido: {response.status_code}. "
        f"Respuesta: {response.text}"
    )