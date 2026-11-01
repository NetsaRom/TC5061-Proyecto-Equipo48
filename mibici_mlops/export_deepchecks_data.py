"""
Exporta los conjuntos de entrenamiento y prueba de P1
para realizar validaciones de calidad con Deepchecks.

No entrena ningún modelo.
"""

from pathlib import Path

from mibici_mlops.modeling.experiment_p1 import prepare_data


def main():
    # Carpeta raíz del proyecto
    root_dir = Path(__file__).resolve().parents[1]

    # Carpeta de salida
    output_dir = root_dir / "data" / "interim"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reutilizar la preparación de datos del modelo P1
    X_train, X_test, *_ = prepare_data()

    # Exportar conjuntos de características
    train_path = output_dir / "deepchecks_X_train.parquet"
    test_path = output_dir / "deepchecks_X_test.parquet"

    X_train.to_parquet(train_path, index=False)
    X_test.to_parquet(test_path, index=False)

    print("\n===== EXPORTACIÓN PARA DEEPCHECKS =====")
    print(f"Entrenamiento: {X_train.shape}")
    print(f"Prueba: {X_test.shape}")
    print(f"Archivo de entrenamiento: {train_path}")
    print(f"Archivo de prueba: {test_path}")
    print("\nExportación completada correctamente.")


if __name__ == "__main__":
    main()