from pathlib import Path
import typer
import polars as pl
import mlflow
import mlflow.sklearn
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Importar las rutas del archivo de configuración del equipo
from mibici_mlops.config import PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()

@app.command()
def entrenar_clustering(
    input_path: Path = PROCESSED_DATA_DIR / "viajes_2026_09.parquet",
    n_clusters: int = typer.Option(4, "--clusters", "-k", help="Número de grupos para K-Means"),
):
    """Entrena un modelo K-Means para agrupar estaciones de MiBici,

    evalúa su rendimiento y registra las métricas en MLflow.
    """
    # 1. Configuración de MLflow
    mlflow.set_tracking_uri("file://" + str(PROJ_ROOT / "mlruns"))
    mlflow.set_experiment("MiBici_Estaciones_Clustering")

    logger.info(f"Cargando datos procesados desde: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo Parquet: {input_path}")
        
    df = pl.read_parquet(input_path)

    # 2. Ingeniería de Características (Feature Engineering por Estación)
    # Filtramos viajes atípicos para no sesgar el comportamiento normal
    df_filtrado = df.filter(pl.col("duracion_atipica") == False)

    logger.info("Calculando métricas agregadas por estación de origen...")
    estaciones_features = (
        df_filtrado.group_by("Origen_Id")
        .agg([
            pl.len().alias("total_viajes_salientes"),
            pl.col("duracion_min").mean().alias("promedio_duracion_min"),
        ])
        .drop_nulls()
    )

    # Convertir a matriz NumPy para Scikit-Learn
    X = estaciones_features.select(["total_viajes_salientes", "promedio_duracion_min"]).to_numpy()
    
    # 3. Escalamiento de Datos (Vital para K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Iniciar tracking del experimento en MLflow
    nombre_run = f"KMeans_K{n_clusters}_{input_path.stem}"
    
    with mlflow.start_run(run_name=nombre_run):
        logger.info(f"Entrenando K-Means con K={n_clusters}...")
        
        # Entrenar Modelo
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = model.fit_predict(X_scaled)

        # 5. Evaluación del Modelo
        logger.info("Evaluando la calidad de los clusters...")
        score_silhouette = silhouette_score(X_scaled, cluster_labels)
        score_davies = davies_bouldin_score(X_scaled, cluster_labels)
        
        logger.info(f"Métricas calculadas -> Silhouette: {score_silhouette:.4f} | Davies-Bouldin: {score_davies:.4f}")

        # 6. Guardar Parámetros y Métricas en MLflow
        mlflow.log_param("algoritmo", "KMeans")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("total_estaciones_procesadas", X.shape[0])
        mlflow.log_param("dataset_origen", input_path.name)

        mlflow.log_metric("silhouette_score", score_silhouette)
        mlflow.log_metric("davies_bouldin_index", score_davies)
        mlflow.log_metric("inertia", model.inertia_)

        # 7. Guardar el modelo y el escalador como artefactos
        mlflow.sklearn.log_model(model, "modelo_kmeans")
        
        logger.success(f"¡Experimento '{nombre_run}' registrado exitosamente en MLflow!")

if __name__ == "__main__":
    app()

