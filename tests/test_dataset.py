
import json
import hashlib
from datetime import datetime, timedelta

import polars as pl
import pytest

from mibici_mlops.dataset import (
    limpiar_viajes,
    leer_estaciones,
    generar_manifiesto
)


def test_limpiar_viajes():
    """Verifica limpieza de duplicados y duración."""

    inicio = datetime(2026, 9, 1, 8, 0)

    df = pl.DataFrame({
        "Viaje_Id": [1, 1, 2],
        "Inicio_del_viaje": [inicio, inicio, inicio],
        "Fin_del_viaje": [
            inicio + timedelta(minutes=60),
            inicio + timedelta(minutes=60),
            inicio + timedelta(minutes=30)
        ],
        "Origen_Id": [10, 10, 20],
        "Destino_Id": [11, 11, 21]
    })

    resultado = limpiar_viajes(df)

    assert resultado.height == 2
    assert resultado["duracion_min"].to_list() == [60.0, 30.0]
    assert resultado["duracion_atipica"].to_list() == [
        False, False
    ]


def test_limpiar_viajes_columnas_faltantes():
    """Verifica que se detecten columnas obligatorias faltantes."""

    df = pl.DataFrame({
        "Viaje_Id": [1],
        "Origen_Id": [10]
    })

    with pytest.raises(ValueError):
        limpiar_viajes(df)


def test_leer_estaciones():
    """Verifica la lectura correcta del catálogo."""

    ruta = pytest.importorskip("pathlib").Path(
        "tests/estaciones_test.csv"
    )

    ruta.write_text(
        "id,name,latitude,longitude,dpcapacity\n"
        "1,Estacion A,20.67,-103.35,20\n"
        "2,Estacion B,20.68,-103.36,25\n",
        encoding="utf-8"
    )

    resultado = leer_estaciones(ruta)

    assert resultado.height == 2
    assert "latitude" in resultado.columns

    ruta.unlink()


def test_leer_estaciones_ids_duplicados(tmp_path):
    """Verifica que se rechacen estaciones duplicadas."""

    ruta = tmp_path / "estaciones.csv"

    ruta.write_text(
        "id,name,latitude,longitude,dpcapacity\n"
        "1,Estacion A,20.67,-103.35,20\n"
        "1,Estacion B,20.68,-103.36,25\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="IDs duplicados"):
        leer_estaciones(ruta)


def test_generar_manifiesto(tmp_path):
    """Verifica la generación correcta del manifiesto."""

    archivo = tmp_path / "datos.txt"
    archivo.write_text("MiBici Equipo 48", encoding="utf-8")

    salida = tmp_path / "manifest.json"

    generar_manifiesto([archivo], salida)

    assert salida.exists()

    with open(salida, encoding="utf-8") as f:
        manifiesto = json.load(f)

    assert manifiesto["equipo"] == 48
    assert len(manifiesto["archivos"]) == 1

    hash_esperado = hashlib.sha256(
        b"MiBici Equipo 48"
    ).hexdigest()

    assert manifiesto["archivos"][0]["sha256"] == hash_esperado


def test_generar_manifiesto_archivo_inexistente(tmp_path):
    """Verifica el manejo de archivos inexistentes."""

    archivo = tmp_path / "inexistente.csv"
    salida = tmp_path / "manifest.json"

    with pytest.raises(FileNotFoundError):
        generar_manifiesto([archivo], salida)
