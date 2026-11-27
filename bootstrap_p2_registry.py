"""Initialize the P2 model registry from the packaged artifact."""

import hashlib
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "mibici-equipo48-p2-riesgo"
MODEL_ALIAS = "recovered"
MODEL_PATH = Path("/app/models/p2")
EXPERIMENT_NAME = "P2_Model_Recovery"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    client = MlflowClient()
    checksum = sha256(MODEL_PATH / "model.skops")

    try:
        version = client.get_model_version_by_alias(
            MODEL_NAME, MODEL_ALIAS
        )
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        version = None

    if version is None:
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(
            run_name="P2_Artifact_Recovery"
        ) as run:
            mlflow.set_tags({
                "question": "P2",
                "stage_of_work": "recovery",
                "source": "repository/models/p2",
                "lineage_status": "historical_lineage_unavailable",
                "original_run_id": "0766e362faf644129dd54d9a2993380c",
                "original_model_id": "m-c390896ac5bb4a9abfb743215b7c55e1",
                "retrained": "false",
            })

            mlflow.log_param("model_sha256", checksum)
            mlflow.log_artifacts(
                str(MODEL_PATH), artifact_path="model"
            )

            version = mlflow.register_model(
                f"runs:/{run.info.run_id}/model",
                MODEL_NAME,
            )

        client.set_model_version_tag(
            MODEL_NAME, version.version,
            "lineage_status", "recovered"
        )
        client.set_model_version_tag(
            MODEL_NAME, version.version,
            "retrained", "false"
        )
        client.set_model_version_tag(
            MODEL_NAME, version.version,
            "promotion_status", "pending_validation"
        )
    else:
        print("Existing registered model found.")

    downloaded = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{MODEL_NAME}/{version.version}"
        )
    )

    if sha256(downloaded / "model.skops") != checksum:
        raise RuntimeError("Model integrity check failed.")

    client.set_registered_model_alias(
        MODEL_NAME, MODEL_ALIAS, version.version
    )

    print(f"Model: {MODEL_NAME}")
    print(f"Version: {version.version}")
    print(f"Alias: {MODEL_ALIAS}")
    print(f"SHA256: {checksum}")
    print("REGISTRY BOOTSTRAP: PASSED")


if __name__ == "__main__":
    main()
