
import pytest
import polars as pl

from mibici_mlops.features import generar_caracteristicas


def crear_datos_prueba():
    """Crea un conjunto pequeño de datos para las pruebas."""

    return pl.DataFrame({
        "Viaje_Id": [1, 2, 3, 4, 5, 6],
        "Inicio_del_viaje": [
            "2026-09-01 07:00:00",
            "2026-09-01 08:00:00",
            "2026-09-01 09:00:00",
            "2026-09-01 10:00:00",
            "2026-09-01 11:00:00",
            "2026-09-01 12:00:00",
        ],
        "Fin_del_viaje": [
            "2026-09-01 07:15:00",
            "2026-09-01 08:20:00",
            "2026-09-01 09:25:00",
            "2026-09-01 10:10:00",
            "2026-09-01 11:30:00",
            "2026-09-01 12:15:00",
        ],
        "Origen_Id": [1, 1, 1, 2, 2, 3],
        "Destino_Id": [2, 3, 2, 1, 3, 1],
        "duracion_min": [15, 20, 25, 10, 30, 15],
        "duracion_atipica": [False, False, False, False, False, False],
    }).with_columns([
        pl.col("Inicio_del_viaje").str.to_datetime(),
        pl.col("Fin_del_viaje").str.to_datetime(),
    ])


def test_generar_caracteristicas(tmp_path):
    """Verifica que se genere correctamente el archivo de características."""

    input_path = tmp_path / "viajes.parquet"
    output_path = tmp_path / "features.parquet"

    crear_datos_prueba().write_parquet(input_path)

    generar_caracteristicas(
        input_path=input_path,
        output_path=output_path
    )

    assert output_path.exists()

    resultado = pl.read_parquet(output_path)

    assert not resultado.is_empty()
    assert "viajes_salientes" in resultado.columns
    assert "promedio_duracion_salida" in resultado.columns
    assert "pct_viajes_matutinos" in resultado.columns
    assert "viajes_entrantes" in resultado.columns
    assert "alta_demanda" in resultado.columns


def test_generar_caracteristicas_demanda(tmp_path):
    """Verifica el cálculo de viajes salientes y demanda por estación."""

    input_path = tmp_path / "viajes.parquet"
    output_path = tmp_path / "features.parquet"

    crear_datos_prueba().write_parquet(input_path)

    generar_caracteristicas(
        input_path=input_path,
        output_path=output_path
    )

    resultado = pl.read_parquet(output_path)

    estacion_1 = resultado.filter(
        pl.col("Origen_Id") == 1
    )

    assert estacion_1["viajes_salientes"][0] == 3
    assert estacion_1["promedio_duracion_salida"][0] == 20.0
    assert estacion_1["pct_viajes_matutinos"][0] == 1.0
    assert estacion_1["alta_demanda"][0] == 1


def test_generar_caracteristicas_archivo_inexistente(tmp_path):
    """Verifica el comportamiento cuando el archivo de entrada no existe."""

    input_path = tmp_path / "no_existe.parquet"
    output_path = tmp_path / "features.parquet"

    with pytest.raises(FileNotFoundError):
        generar_caracteristicas(
            input_path=input_path,
            output_path=output_path
        )


def test_generar_caracteristicas_sin_viajes_validos(tmp_path):
    """Verifica el comportamiento cuando todos los viajes son atípicos."""

    input_path = tmp_path / "viajes.parquet"
    output_path = tmp_path / "features.parquet"

    df = crear_datos_prueba().with_columns(
        pl.lit(True).alias("duracion_atipica")
    )

    df.write_parquet(input_path)

    with pytest.raises(
        ValueError,
        match="No hay viajes válidos"
    ):
        generar_caracteristicas(
            input_path=input_path,
            output_path=output_path
        )
