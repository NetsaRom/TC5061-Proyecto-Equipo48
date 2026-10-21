"""Punto de entrada unificado para entrenamientos reproducibles con MLflow."""

import os

import typer

from mibici_mlops.config import (
    FECHA_FIN_EXCLUSIVA,
    FECHA_INICIO,
    MAX_DEPTH,
    N_CLUSTERS,
    N_ESTIMATORS,
    RANDOM_SEED,
    TEST_SIZE,
)

app = typer.Typer(help="Entrena modelos MiBici mediante MLflow Projects.")


@app.command()
def train(
    question: str = typer.Option(
        ...,
        "--question",
        help="Pregunta del proyecto: p1, p2 o p3.",
    ),
    model: str = typer.Option(
        "default",
        "--model",
        help="Modelo a utilizar; default selecciona el modelo del proyecto.",
    ),
    n_estimators: int = typer.Option(N_ESTIMATORS, "--n-estimators"),
    max_depth: int = typer.Option(MAX_DEPTH, "--max-depth"),
    n_clusters: int = typer.Option(N_CLUSTERS, "--n-clusters"),
    test_size: float = typer.Option(TEST_SIZE, "--test-size"),
) -> None:
    """Despacha la ejecución al entrenamiento correspondiente."""

    question = question.lower()
    if question not in {"p1", "p2", "p3"}:
        raise typer.BadParameter("Usa p1, p2 o p3 para --question.")

    if model != "default":
        expected = {
            "p1": "RandomForestRegressor",
            "p2": "RandomForestClassifier",
            "p3": "KMeans",
        }[question]
        if model != expected:
            raise typer.BadParameter(
                f"Para {question}, el modelo admitido es {expected}."
            )

    if n_estimators < 1:
        raise typer.BadParameter("--n-estimators debe ser positivo.")
    if max_depth < 1:
        raise typer.BadParameter("--max-depth debe ser positivo.")
    if n_clusters < 2:
        raise typer.BadParameter("--n-clusters debe ser al menos 2.")
    if not 0 < test_size < 1:
        raise typer.BadParameter("--test-size debe estar entre 0 y 1.")

    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "mibici-equipo48")

    if question == "p1":
        from mibici_mlops.modeling import train_p1_regression as training

        training.N_ESTIMATORS = n_estimators
        training.MAX_DEPTH = max_depth
        training.RANDOM_STATE = RANDOM_SEED
        training.START_DATE = f"{FECHA_INICIO}-01"
        training.END_DATE = FECHA_FIN_EXCLUSIVA
        training.run_experiment()

    elif question == "p2":
        from mibici_mlops.modeling.train_p2_classification import (
            entrenar_clasificador,
        )

        entrenar_clasificador(
            n_estimators=n_estimators,
            test_size=test_size,
        )

    else:
        from mibici_mlops.modeling import train_p3_clustering as training

        training.N_CLUSTERS = n_clusters
        training.RANDOM_STATE = RANDOM_SEED
        training.START_DATE = f"{FECHA_INICIO}-01"
        training.END_DATE = FECHA_FIN_EXCLUSIVA
        training.run_experiment()


if __name__ == "__main__":
    app()
