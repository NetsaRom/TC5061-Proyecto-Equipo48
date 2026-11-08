import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests


# Dirección de FastAPI.
# Puede modificarse mediante la variable de entorno API_BASE_URL.
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

# Carpeta donde se guardará la evidencia.
EVIDENCE_DIRECTORY = Path("reports/evidencias")

# Tiempo máximo de espera para cada solicitud.
TIMEOUT_SECONDS = 15

# Aquí se almacenarán los resultados de las pruebas.
checks = []


def run_test(
    name: str,
    method: str,
    path: str,
    expected_status: int = 200,
    **request_arguments,
) -> None:
    """
    Ejecuta una solicitud HTTP contra FastAPI y almacena el resultado.

    Args:
        name: Nombre de la prueba.
        method: Método HTTP, por ejemplo GET o POST.
        path: Endpoint que se probará.
        expected_status: Código HTTP esperado.
        request_arguments: Argumentos adicionales para requests,
            por ejemplo json, headers o params.
    """

    url = f"{API_BASE_URL}{path}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=TIMEOUT_SECONDS,
            **request_arguments,
        )

        try:
            response_body = response.json()
        except ValueError:
            response_body = response.text

        test_passed = response.status_code == expected_status

        checks.append(
            {
                "prueba": name,
                "metodo": method,
                "endpoint": path,
                "url": url,
                "status_esperado": expected_status,
                "status_obtenido": response.status_code,
                "respuesta": response_body,
                "resultado": "APROBADA" if test_passed else "NO APROBADA",
                "ok": test_passed,
            }
        )

    except requests.Timeout:
        checks.append(
            {
                "prueba": name,
                "metodo": method,
                "endpoint": path,
                "url": url,
                "resultado": "NO APROBADA",
                "ok": False,
                "error": (
                    f"La solicitud superó el tiempo máximo "
                    f"de {TIMEOUT_SECONDS} segundos."
                ),
            }
        )

    except requests.ConnectionError:
        checks.append(
            {
                "prueba": name,
                "metodo": method,
                "endpoint": path,
                "url": url,
                "resultado": "NO APROBADA",
                "ok": False,
                "error": (
                    "No fue posible establecer comunicación con FastAPI. "
                    "Verifica que el contenedor mibici-api-p2 esté activo."
                ),
            }
        )

    except requests.RequestException as error:
        checks.append(
            {
                "prueba": name,
                "metodo": method,
                "endpoint": path,
                "url": url,
                "resultado": "NO APROBADA",
                "ok": False,
                "error": str(error),
            }
        )


def main() -> int:
    """
    Ejecuta las pruebas de extremo a extremo y genera la evidencia JSON.

    Returns:
        0 cuando todas las pruebas son aprobadas.
        1 cuando una o más pruebas no son aprobadas.
    """

    EVIDENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRUEBAS DE EXTREMO A EXTREMO")
    print("=" * 70)
    print(f"API evaluada: {API_BASE_URL}")
    print()

    # Prueba 1: verificar disponibilidad de FastAPI y MLflow.
    run_test(
        name="Comunicación FastAPI - MLflow",
        method="GET",
        path="/health",
        expected_status=200,
    )

    # Prueba 2: realizar una inferencia válida.
    valid_payload = {
        "promedio_duracion_salida": 9.2547,
        "pct_viajes_matutinos": 0.2379,
        "viajes_entrantes": 1452,
    }

    run_test(
        name="Predicción con datos válidos",
        method="POST",
        path="/predict",
        expected_status=200,
        json=valid_payload,
        headers={"Content-Type": "application/json"},
    )

    # Prueba 3: comprobar el rechazo de valores fuera del contrato.
    invalid_payload = {
        "promedio_duracion_salida": -10,
        "pct_viajes_matutinos": 1.5,
        "viajes_entrantes": -5,
    }

    run_test(
        name="Rechazo de datos inválidos",
        method="POST",
        path="/predict",
        expected_status=422,
        json=invalid_payload,
        headers={"Content-Type": "application/json"},
    )

    all_tests_passed = all(check["ok"] for check in checks)

    report = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "api_evaluada": API_BASE_URL,
        "total_pruebas": len(checks),
        "pruebas_aprobadas": sum(check["ok"] for check in checks),
        "pruebas_no_aprobadas": sum(
            not check["ok"] for check in checks
        ),
        "resultado": (
            "APROBADO"
            if all_tests_passed
            else "NO APROBADO"
        ),
        "pruebas": checks,
    }

    report_path = EVIDENCE_DIRECTORY / "e2e_resultados.json"

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for check in checks:
        symbol = "OK" if check["ok"] else "ERROR"

        print(
            f"[{symbol}] {check['prueba']}: "
            f"{check['resultado']}"
        )

        if "status_obtenido" in check:
            print(
                f"       HTTP esperado: {check['status_esperado']} | "
                f"HTTP obtenido: {check['status_obtenido']}"
            )

        if "error" in check:
            print(f"       Error: {check['error']}")

    print()
    print("=" * 70)
    print(f"RESULTADO GENERAL: {report['resultado']}")
    print(f"Evidencia generada: {report_path}")
    print("=" * 70)

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())