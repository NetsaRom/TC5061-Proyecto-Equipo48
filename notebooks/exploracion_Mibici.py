
from pathlib import Path
import polars as pl

# 1. Ruta del archivo original
ruta = Path("data/raw/datos_abiertos_2026_09.csv")

# 2. Leer los datos con Polars
df = pl.read_csv(
    ruta,
    null_values=["NULL", "NA", ""],
    try_parse_dates=True,
    infer_schema_length=10000,
)

# 3. Mostrar dimensiones
print("DIMENSIONES DEL DATASET")
print(f"Filas: {df.height:,}")
print(f"Columnas: {df.width}")

# 4. Mostrar nombres y tipos de columnas
print("\nESQUEMA DEL DATASET")
print(df.schema)

# 5. Mostrar primeros cinco registros
print("\nPRIMEROS 5 REGISTROS")
print(df.head(5))

# 6. Contar valores faltantes
print("\nVALORES FALTANTES POR COLUMNA")
print(df.null_count())


# 7. Verificar identificadores de viaje duplicados
duplicados = df.select(
    pl.col("Viaje_Id").is_duplicated().sum()
).item()

print("\nVIAJES DUPLICADOS")
print(f"IDs duplicados: {duplicados:,}")

# 8. Revisar rango de fechas
print("\nRANGO DE FECHAS")
print(
    df.select(
        pl.col("Inicio_del_viaje").min().alias("inicio_min"),
        pl.col("Inicio_del_viaje").max().alias("inicio_max"),
        pl.col("Fin_del_viaje").max().alias("fin_max"),
    )
)

# 9. Calcular duración de los viajes en minutos
df = df.with_columns(
    (
        (pl.col("Fin_del_viaje") - pl.col("Inicio_del_viaje"))
        .dt.total_seconds() / 60
    ).alias("duracion_min")
)

print("\nESTADÍSTICAS DE DURACIÓN")
print(df.select("duracion_min").describe())

# 10. Identificar viajes con duración no positiva
duraciones_invalidas = df.filter(
    pl.col("duracion_min") <= 0
).height

print(f"\nViajes con duración <= 0: {duraciones_invalidas:,}")

# 11. Contar estaciones únicas
print("\nESTACIONES")
print(
    "Estaciones de origen:",
    df["Origen_Id"].n_unique()
)
print(
    "Estaciones de destino:",
    df["Destino_Id"].n_unique()
)


# 12. Analizar duplicados

print("\nANÁLISIS DE DUPLICADOS")

# Filas completamente idénticas
duplicados_exactos = df.height - df.unique().height

print(f"Filas duplicadas exactas: {duplicados_exactos:,}")

# Identificadores de viaje que aparecen más de una vez
ids_repetidos = (
    df.group_by("Viaje_Id")
    .agg(pl.len().alias("frecuencia"))
    .filter(pl.col("frecuencia") > 1)
    .sort("frecuencia", descending=True)
)

print(f"IDs diferentes repetidos: {ids_repetidos.height:,}")
print("\nEjemplos de IDs repetidos:")
print(ids_repetidos.head(10))

# 13. Investigar duraciones
print("\nANÁLISIS DE DURACIONES")

print(
    "Duración igual a cero:",
    df.filter(pl.col("duracion_min") == 0).height
)

print(
    "Duración negativa:",
    df.filter(pl.col("duracion_min") < 0).height
)

print(
    "Duración mayor a 60 minutos:",
    df.filter(pl.col("duracion_min") > 60).height
)

print(
    "Duración mayor a 24 horas:",
    df.filter(pl.col("duracion_min") > 1440).height
)
