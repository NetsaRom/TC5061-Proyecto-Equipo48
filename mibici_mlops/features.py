from pathlib import Path
import typer
import polars as pl
from loguru import logger
from mibici_mlops.config import PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def generar_caracteristicas(
    input_path: Path = PROCESSED_DATA_DIR / "viajes_2026_09.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "features_estaciones.parquet",
):
    """Genera variables de ingeniería de características por estación."""
    logger.info(f"Cargando datos para feature engineering: {input_path}")
    df = pl.read_parquet(input_path).filter(pl.col("duracion_atipica") == False)

    # Extraer componentes de tiempo para enriquecer las características
    df = df.with_columns([
        pl.col("Inicio_del_viaje").dt.hour().alias("hora_inicio"),
    ])

    logger.info("Calculando métricas avanzadas por estación...")
    # Agrupación por Origen
    origen_features = df.group_by("Origen_Id").agg([
        pl.len().alias("viajes_salientes"),
        pl.col("duracion_min").mean().alias("promedio_duracion_salida"),
        (pl.col("hora_inicio").filter((pl.col("hora_inicio") >= 6) & (pl.col("hora_inicio") <= 9)).len() / pl.len()).alias("pct_viajes_matutinos"),
    ])

    # Agrupación por Destino
    destino_features = df.group_by("Destino_Id").agg([
        pl.len().alias("viajes_entrantes"),
    ])

    # Fusión de características
    features = origen_features.join(
        destino_features, left_on="Origen_Id", right_on="Destino_Id", how="full"
    ).fill_null(0)

    # Crear una variable objetivo sintética para la clasificación (Estación de alta demanda)
    mediana_salientes = features["viajes_salientes"].median()
    features = features.with_columns([
        (pl.col("viajes_salientes") > mediana_salientes).cast(pl.Int32).alias("alta_demanda")
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(output_path)
    logger.success(f"Características guardadas exitosamente en: {output_path}")

if __name__ == "__main__":
    app()
