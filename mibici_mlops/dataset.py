
from pathlib import Path

import polars as pl
import typer
from loguru import logger

from mibici_mlops.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def leer_viajes(ruta: Path) -> pl.DataFrame:
    """Lee el archivo CSV original de MiBici."""

    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    return pl.read_csv(
        ruta,
        null_values=["NULL", "NA", ""],
        try_parse_dates=True,
        infer_schema_length=10000,
    )


def limpiar_viajes(df: pl.DataFrame) -> pl.DataFrame:
    """Elimina duplicados exactos y prepara los viajes."""

    columnas = [
        "Viaje_Id",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]

    faltantes = set(columnas) - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(faltantes)}"
        )

    # Eliminar registros completamente duplicados
    df_limpio = df.unique(maintain_order=True)

    # Conservar las columnas operativas
    df_limpio = df_limpio.select(columnas)

    # Calcular duración en minutos
    df_limpio = df_limpio.with_columns(
        (
            (
                pl.col("Fin_del_viaje")
                - pl.col("Inicio_del_viaje")
            ).dt.total_seconds()
            / 60
        ).alias("duracion_min")
    )

    # Marcar duraciones anómalas sin eliminarlas
    df_limpio = df_limpio.with_columns(
        (
            (pl.col("duracion_min") <= 0)
            | (pl.col("duracion_min") > 1440)
        ).alias("duracion_atipica")
    )

    return df_limpio


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "datos_abiertos_2026_09.csv",
    output_path: Path = PROCESSED_DATA_DIR / "viajes_2026_09.parquet",
):
    """Procesa viajes MiBici y genera un archivo Parquet."""

    logger.info(f"Leyendo archivo: {input_path}")

    df = leer_viajes(input_path)
    logger.info(f"Registros originales: {df.height:,}")

    df_limpio = limpiar_viajes(df)
    logger.info(f"Registros procesados: {df_limpio.height:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_limpio.write_parquet(output_path)

    logger.success(f"Archivo generado: {output_path}")


if __name__ == "__main__":
    app()
