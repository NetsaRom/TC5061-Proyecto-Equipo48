
from pathlib import Path

import polars as pl
from loguru import logger

from mibici_mlops.config import PROCESSED_DATA_DIR


def generar_features_clasificacion():
    """Genera características históricas por estación para clasificación."""

    archivos = [
        PROCESSED_DATA_DIR / "viajes_2026_07.parquet",
        PROCESSED_DATA_DIR / "viajes_2026_08.parquet",
        PROCESSED_DATA_DIR / "viajes_2026_09.parquet",
    ]

    for archivo in archivos:
        if not archivo.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {archivo}"
            )

    logger.info("Cargando archivos de viajes...")

    viajes = pl.concat(
        [pl.read_parquet(archivo) for archivo in archivos],
        how="vertical",
    )

    # Preparar información temporal
    viajes = viajes.with_columns(
        pl.col("Inicio_del_viaje").dt.hour().alias("hora_inicio")
    )

    # Características de los viajes que salen de cada estación
    salidas = (
        viajes.group_by("Origen_Id")
        .agg(
            pl.len().alias("viajes_salientes"),
            pl.col("duracion_min")
            .mean()
            .alias("promedio_duracion_salida"),
            (
                (
                    (pl.col("hora_inicio") >= 6)
                    & (pl.col("hora_inicio") < 12)
                ).mean()
                * 100
            ).alias("pct_viajes_matutinos"),
        )
        .rename({"Origen_Id": "estacion"})
    )

    # Cantidad de viajes que llegan a cada estación
    entradas = (
        viajes.group_by("Destino_Id")
        .agg(
            pl.len().alias("viajes_entrantes")
        )
        .rename({"Destino_Id": "estacion"})
    )

    # Combinar características
    features = salidas.join(
        entradas,
        on="estacion",
        how="left",
    ).with_columns(
        pl.col("viajes_entrantes").fill_null(0)
    )

    # Etiqueta exploratoria: estaciones con salidas iguales
    # o superiores a la mediana de viajes salientes.
    # Esto identifica demanda relativa, no riesgo real de desabasto.
    mediana = features["viajes_salientes"].median()

    features = features.with_columns(
        (pl.col("viajes_salientes") >= mediana)
        .cast(pl.Int8)
        .alias("alta_demanda")
    )

    # Validar que existan ambas clases
    clases = features["alta_demanda"].n_unique()

    if clases != 2:
        raise ValueError(
            "No se generaron las dos clases necesarias "
            "para entrenar el clasificador."
        )

    # Guardar archivo de características
    output_path = (
        PROCESSED_DATA_DIR / "features_estaciones.parquet"
    )

    features.write_parquet(output_path)

    logger.success(
        f"Archivo generado: {output_path}"
    )
    logger.info(f"Estaciones: {features.height}")
    logger.info(
        f"Distribución de clases: "
        f"{features.group_by('alta_demanda').len().sort('alta_demanda')}"
    )

    return output_path


if __name__ == "__main__":
    generar_features_clasificacion()
