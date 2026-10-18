import hashlib

from mibici_mlops.dataset import calcular_sha256

import json

from mibici_mlops.dataset import generar_manifiesto

import pytest


def test_calcular_sha256(tmp_path):
    """Verifica que el hash SHA-256 se calcula correctamente."""

    # Crear un archivo temporal
    archivo = tmp_path / "prueba.txt"
    archivo.write_text("MiBici Equipo 48", encoding="utf-8")

    # Calcular el hash esperado
    esperado = hashlib.sha256(
        b"MiBici Equipo 48"
    ).hexdigest()

    # Calcular el hash con nuestra función
    resultado = calcular_sha256(archivo)

    # Comprobar que coinciden
    assert resultado == esperado
    assert len(resultado) == 64

def test_generar_manifiesto(tmp_path):
    """Verifica que el manifiesto JSON se genera correctamente."""

    archivo = tmp_path / "viajes.csv"
    archivo.write_text(
        "id,origen\n1,10\n2,20\n",
        encoding="utf-8",
    )

    ruta_manifiesto = tmp_path / "manifest.json"

    generar_manifiesto(
        rutas_archivos=[archivo],
        ruta_salida=ruta_manifiesto,
    )

    manifiesto = json.loads(
        ruta_manifiesto.read_text(encoding="utf-8")
    )

    assert manifiesto["equipo"] == 48
    assert manifiesto["version_manifiesto"] == "1.0"
    assert len(manifiesto["archivos"]) == 1

    datos = manifiesto["archivos"][0]

    assert datos["nombre"] == "viajes.csv"
    assert datos["tamano_bytes"] == archivo.stat().st_size
    assert datos["sha256"] == calcular_sha256(archivo)
    assert datos["registros"] is None

def test_generar_manifiesto_archivo_inexistente(tmp_path):
    """Verifica que se detecten archivos inexistentes."""

    archivo_inexistente = tmp_path / "no_existe.csv"
    ruta_manifiesto = tmp_path / "manifest.json"

    with pytest.raises(FileNotFoundError):
        generar_manifiesto(
            rutas_archivos=[archivo_inexistente],
            ruta_salida=ruta_manifiesto,
        )

    assert not ruta_manifiesto.exists()