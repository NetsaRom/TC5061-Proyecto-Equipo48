"""
Registro y documentación de la versión alternativa del modelo P2.

Responsabilidades:
- Validar la corrida seleccionada.
- Registrar el modelo bajo el mismo nombre existente.
- Evitar registrar dos veces el mismo Run ID.
- Documentar estrategia, parámetros y métricas.
- Validar la carga de la versión registrada.
- No modificar automáticamente el alias champion.
"""

import os
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

# 1. Configuración

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_NAME = "mibici-equipo48-p2-riesgo"

# Nueva corrida seleccionada mediante búsqueda de hiperparámetros.
RUN_ID = os.getenv(
    "P2_V2_RUN_ID",
    "01a56c56ea1e4b9f9ff9eb0429166a89",
)

MODEL_ARTIFACT = "model"

EXPECTED_QUESTION = "P2"
EXPECTED_STAGE = "candidate"


mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

client = MlflowClient()


# 2. Funciones auxiliares

def validate_run():
    """Valida que la corrida corresponda al candidato P2 esperado."""

    run = client.get_run(RUN_ID)

    if run.info.status != "FINISHED":
        raise RuntimeError(
            f"La corrida {RUN_ID} no terminó correctamente. "
            f"Estado encontrado: {run.info.status}"
        )

    question = run.data.tags.get("question")
    stage_of_work = run.data.tags.get("stage_of_work")

    if question != EXPECTED_QUESTION:
        raise ValueError(
            f"La corrida {RUN_ID} no está identificada como P2. "
            f"Valor encontrado: {question}"
        )

    if stage_of_work != EXPECTED_STAGE:
        raise ValueError(
            "La corrida no está identificada como candidata de búsqueda. "
            f"stage_of_work encontrado: {stage_of_work}"
        )

    return run


def validate_model_artifact():
    """
    Verifica que la corrida tenga un modelo registrado.

    En MLflow 3, el modelo puede estar asociado con un LoggedModel.
    MLflow puede resolver automáticamente este modelo durante el registro.
    """

    artifacts = client.list_artifacts(RUN_ID, MODEL_ARTIFACT)

    mlmodel_found = any(
        artifact.path == f"{MODEL_ARTIFACT}/MLmodel"
        for artifact in artifacts
    )

    if not mlmodel_found:
        print(
            "Advertencia: no se encontró model/MLmodel como artefacto "
            "tradicional del run."
        )
        print(
            "MLflow 3 puede resolver el modelo mediante el LoggedModel "
            "asociado con la corrida."
        )


def get_existing_version():
    """Busca si el Run ID ya fue registrado como una versión del modelo."""

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    return next(
        (
            version
            for version in versions
            if version.run_id == RUN_ID
        ),
        None,
    )


def register_model_version():
    """
    Registra el modelo únicamente si el Run ID todavía no tiene
    una versión asociada.
    """

    existing_version = get_existing_version()

    if existing_version is not None:
        print(
            f"El Run ID ya corresponde a la versión "
            f"{existing_version.version}. No se creará otra versión."
        )

        return existing_version.version

    model_uri = f"runs:/{RUN_ID}/{MODEL_ARTIFACT}"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    return registered.version


def document_model_version(run, version_number):
    """Documenta trazabilidad, estrategia, parámetros y métricas."""

    client.update_registered_model(
        name=MODEL_NAME,
        description=(
            "Versión alternativa del modelo P2 generada mediante modificación "
            "de hiperparámetros. Se utilizó RandomForestClassifier con "
            "n_estimators=50, max_depth=10, class_weight='balanced' y "
            "random_state=42. "
            "Procede de la corrida P2_candidate_1. "
            f"Run ID: {RUN_ID}. "
            "La configuración fue evaluada sobre el conjunto de prueba y "
            "se mantuvo como candidata para comparación con el modelo champion."
            ),
    )

    selected_metrics = {
        key: value
        for key, value in run.data.metrics.items()
        if key in {
            "val_accuracy",
            "val_f1_score",
            "val_roc_auc",
            "training_time_seconds",
            "inference_time_seconds",
        }
    }

    metric_text = ", ".join(
        f"{key}={value:.6f}"
        for key, value in sorted(selected_metrics.items())
    )

    description = (
        "Versión alternativa generada mediante búsqueda de "
        "hiperparámetros del modelo P2. "
        "Procede de la corrida P2_search_candidate_1. "
        f"Run ID: {RUN_ID}. "
        f"Métricas registradas: {metric_text}. "
        "Los parámetros y artefactos completos están disponibles "
        "en la corrida de MLflow."
    )

    client.update_model_version(
        name=MODEL_NAME,
        version=version_number,
        description=description,
    )

    version_tags = {
        "question": "P2",
        "model_family": run.data.tags.get(
            "model_family",
            "RandomForestClassifier",
        ),
        "strategy": "hyperparameter_optimization",
        "candidate_run": "P2_search_candidate_1",
        "run_id": RUN_ID,
        "promotion_status": "candidate",
    }

    data_version = run.data.tags.get("data_version")

    if data_version:
        version_tags["data_version"] = data_version

    git_commit = run.data.tags.get("git_commit")

    if git_commit:
        version_tags["git_commit"] = git_commit

    for key, value in version_tags.items():
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=version_number,
            key=key,
            value=str(value),
        )


def validate_registered_model(version_number):
    """Comprueba que la versión registrada pueda cargarse."""

    model_uri = f"models:/{MODEL_NAME}/{version_number}"

    mlflow.pyfunc.load_model(model_uri)

    return model_uri


def get_champion_version():
    """Consulta qué versión tiene actualmente el alias champion."""

    try:
        champion = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="champion",
        )

        return champion.version

    except Exception:
        return None


def print_run_evidence(run, version_number, model_uri):
    """Muestra parámetros, métricas y resultado del registro."""

    print("\n" + "=" * 70)
    print("EVIDENCIA DEL REGISTRO DE P2 V2")
    print("=" * 70)

    print(f"Modelo registrado : {MODEL_NAME}")
    print(f"Versión generada  : {version_number}")
    print(f"Run ID            : {RUN_ID}")
    print(f"URI del modelo    : {model_uri}")
    print(f"Estado del run    : {run.info.status}")
    print("Estrategia        : búsqueda de hiperparámetros")

    print("\nParámetros del entrenamiento:")

    if run.data.params:
        for key, value in sorted(run.data.params.items()):
            print(f"  {key}: {value}")
    else:
        print("  No se encontraron parámetros registrados.")

    print("\nMétricas del entrenamiento:")

    if run.data.metrics:
        for key, value in sorted(run.data.metrics.items()):
            print(f"  {key}: {value}")
    else:
        print("  No se encontraron métricas registradas.")

    champion_version = get_champion_version()

    print("\nEstado del alias champion:")

    if champion_version is None:
        print("  No existe una versión con el alias champion.")
    else:
        print(f"  champion apunta a la versión {champion_version}")

    if str(champion_version) == str(version_number):
        print("  La nueva versión es actualmente champion.")
    else:
        print(
            "  La nueva versión permanece como candidata y "
            "no modificó champion."
        )

    print("\nValidación:")
    print("  Carga directa de la versión registrada: correcta.")
    print("=" * 70)



# 3. Ejecución principal

def main():
    run = validate_run()

    validate_model_artifact()

    version_number = register_model_version()

    document_model_version(
        run=run,
        version_number=version_number,
    )

    model_uri = validate_registered_model(
        version_number=version_number,
    )

    print_run_evidence(
        run=run,
        version_number=version_number,
        model_uri=model_uri,
    )


if __name__ == "__main__":
    main()