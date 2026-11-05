
import polars as pl
import json

# Leer características generadas por MiBici
df = pl.read_parquet(
    "data/processed/features_estaciones.parquet"
)

# Seleccionar tres registros reales
columnas = [
    "promedio_duracion_salida",
    "pct_viajes_matutinos",
    "viajes_entrantes",
]

datos = (
    df.select(columnas)
    .drop_nulls()
    .head(3)
)

# Preparar formato compatible con MLflow Serving
payload = {
    "dataframe_split": {
        "columns": columnas,
        "data": datos.to_numpy().tolist()
    }
}

# Guardar JSON
with open("input.json", "w", encoding="utf-8") as archivo:
    json.dump(payload, archivo, indent=4)

print("Archivo input.json generado correctamente:")
print(json.dumps(payload, indent=4))
