from pathlib import Path
import typer
import polars as pl
import mlflow
import mlflow.sklearn
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mibici_mlops.config import PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()

@app.command()
def entrenar_regresor(
    input_path: Path = PROCESSED_DATA_DIR / "features_estaciones.parquet",
    n_estimators: int = typer.Option(100, "--estimators", "-e"),
):
    """Entrena un regresor RandomForest para predecir el número de viajes salientes."""
    mlflow.set_tracking_uri("file://" + str(PROJ_ROOT / "mlruns"))
    mlflow.set_experiment("MiBici_Regresion_Viajes")

    df = pl.read_parquet(input_path)
    
    # Objetivo: Predecir volumen continuo de viajes salientes
    X = df.select(["promedio_duracion_salida", "pct_viajes_matutinos", "viajes_entrantes"]).to_numpy()
    y = df["viajes_salientes"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name=f"RandomForest_Reg_E{n_estimators}"):
        logger.info("Entrenando modelo de regresión...")
        reg = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        reg.fit(X_train, y_train)

        preds = reg.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        logger.info(f"Evaluación Regresión -> MAE: {mae:.2f} | MSE: {mse:.2f} | R2 Score: {r2:.4f}")

        # MLflow Logs
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(reg, "modelo_regresor")
        
        logger.success("¡Modelo de regresión registrado en MLflow!")

if __name__ == "__main__":
    app()
