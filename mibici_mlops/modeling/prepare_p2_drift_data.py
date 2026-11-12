
from pathlib import Path

import polars as pl
from loguru import logger

from mibici_mlops.config import PROCESSED_DATA_DIR


FEATURES = [
    "promedio_duracion_salida",
    "pct_viajes_matutinos",
    "viajes_entrantes",
]


def preparar_features_mes(periodo: str) -> pl.DataFrame:
    """Calcula las características originales de P2 para un mes."""

    ruta = PROCESSED_DATA_DIR / f"viajes_{periodo}.parquet"

    if not ruta.exists():
        raise FileNotFoundError(f"No existe: {ruta}")

    viajes = pl.read_parquet(ruta)

    columnas_requeridas = [
        "Inicio_del_viaje",
        "Origen_Id",
        "Destino_Id",
        "duracion_min",
    ]

    faltantes = set(columnas_requeridas) - set(viajes.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes: {faltantes}")

    viajes = viajes.with_columns(
        pl.col("Inicio_del_viaje").dt.hour().alias("hora_inicio")
    )

    salidas = (
        viajes.group_by("Origen_Id")
        .agg(
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

    entradas = (
        viajes.group_by("Destino_Id")
        .agg(
            pl.len().alias("viajes_entrantes")
        )
        .rename({"Destino_Id": "estacion"})
    )

    features = (
        salidas.join(entradas, on="estacion", how="left")
        .with_columns(pl.col("viajes_entrantes").fill_null(0))
        .sort("estacion")
    )

    return features


def main() -> None:
    """Genera los datasets Reference y Current para Data Drift P2."""

    # Reference: características originales de entrenamiento (julio-septiembre).
    ruta_reference_original = PROCESSED_DATA_DIR / "features_estaciones.parquet"

    if not ruta_reference_original.exists():
        raise FileNotFoundError(
            f"No existe: {ruta_reference_original}. "
            "Ejecuta primero: python -m mibici_mlops.modeling.features_clasificacion"
        )

    reference = (
        pl.read_parquet(ruta_reference_original)
        .select(["estacion", *FEATURES])
        .sort("estacion")
    )

    # Current: características del periodo posterior (octubre).
    current = preparar_features_mes("2026_10")

    if reference.schema != current.schema:
        raise ValueError("Los esquemas Reference y Current no coinciden")

    for nombre, df in [("Reference", reference), ("Current", current)]:
        if df.select(FEATURES).null_count().sum_horizontal().item() != 0:
            raise ValueError(f"{nombre} contiene valores nulos")

    salida = Path("data/interim")
    salida.mkdir(parents=True, exist_ok=True)

    ruta_reference = salida / "p2_drift_reference.parquet"
    ruta_current = salida / "p2_drift_current.parquet"

    reference.write_parquet(ruta_reference)
    current.write_parquet(ruta_current)

    logger.success(f"Reference: {ruta_reference} ({reference.height} estaciones)")
    logger.success(f"Current: {ruta_current} ({current.height} estaciones)")


if __name__ == "__main__":
    main()
