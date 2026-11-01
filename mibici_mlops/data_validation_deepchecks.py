"""
Validación de calidad de datos para MiBici Guadalajara.

Verificaciones:
1. Tipos de datos.
2. Valores faltantes.
3. Registros duplicados.
4. Data drift entre entrenamiento y prueba.

Herramienta: Deepchecks 0.19.1.
"""

from pathlib import Path

import pandas as pd

from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    PercentOfNulls,
    DataDuplicates,
    FeatureDrift,
)

# Directorio principal del proyecto
ROOT_DIR = Path(__file__).resolve().parents[1]

# Archivos generados a partir del experimento P1
TRAIN_PATH = ROOT_DIR / "data/interim/deepchecks_X_train.parquet"
TEST_PATH = ROOT_DIR / "data/interim/deepchecks_X_test.parquet"


def load_data():
    """Carga los conjuntos de entrenamiento y prueba."""

    X_train = pd.read_parquet(TRAIN_PATH)
    X_test = pd.read_parquet(TEST_PATH)

    print("\n===== DATOS MIBICI =====")
    print(f"Entrenamiento: {X_train.shape}")
    print(f"Prueba: {X_test.shape}")

    return X_train, X_test

def validate_data_types(X_train, X_test):
    """Muestra los tipos de datos de entrenamiento y prueba."""

    print("\n===== VALIDACIÓN DE TIPOS DE DATOS =====")

    comparison = pd.DataFrame({
        "Entrenamiento": X_train.dtypes.astype(str),
        "Prueba": X_test.dtypes.astype(str),
    })

    comparison["Coinciden"] = (
        comparison["Entrenamiento"] == comparison["Prueba"]
    )

    print(comparison.to_string())

    if comparison["Coinciden"].all():
        print("\nRESULTADO: Los tipos de datos coinciden.")
    else:
        print("\nADVERTENCIA: Se encontraron diferencias de tipos.")

    return comparison

def validate_missing_values(X_train, X_test):
    """Identifica valores faltantes en entrenamiento y prueba."""

    print("\n===== VALIDACIÓN DE VALORES FALTANTES =====")

    missing = pd.DataFrame({
        "Entrenamiento": X_train.isnull().sum(),
        "Prueba": X_test.isnull().sum(),
    })

    missing["Total"] = (
        missing["Entrenamiento"] + missing["Prueba"]
    )

    print(missing.to_string())

    total_missing = missing["Total"].sum()

    if total_missing == 0:
        print("\nRESULTADO: No se encontraron valores faltantes.")
    else:
        print(f"\nADVERTENCIA: Se encontraron {total_missing:,} valores faltantes.")

    return missing

def validate_duplicates(X_train, X_test):
    """Detecta filas idénticas en entrenamiento y prueba."""

    print("\n===== VALIDACIÓN DE DUPLICADOS =====")

    train_duplicates = X_train.duplicated().sum()
    test_duplicates = X_test.duplicated().sum()

    train_percentage = train_duplicates / len(X_train) * 100
    test_percentage = test_duplicates / len(X_test) * 100

    print(
        f"Entrenamiento: {train_duplicates:,} filas duplicadas "
        f"({train_percentage:.2f} %)"
    )

    print(
        f"Prueba: {test_duplicates:,} filas duplicadas "
        f"({test_percentage:.2f} %)"
    )

    return {
        "train_duplicates": int(train_duplicates),
        "test_duplicates": int(test_duplicates),
    }


def validate_nulls_deepchecks(X_train, X_test):
    """Analiza valores faltantes utilizando Deepchecks."""

    print("\n===== DEEPCHECKS: VALORES FALTANTES =====")

    train_dataset = Dataset(X_train, cat_features=["estacion"])
    test_dataset = Dataset(X_test, cat_features=["estacion"])

    check = PercentOfNulls()

    print("\n--- Entrenamiento ---")
    train_result = check.run(train_dataset)
    print(train_result.value)

    print("\n--- Prueba ---")
    test_result = check.run(test_dataset)
    print(test_result.value)

    return train_result, test_result

def validate_duplicates_deepchecks(X_train, X_test):
    """Detecta registros duplicados utilizando Deepchecks."""

    print("\n===== DEEPCHECKS: DUPLICADOS =====")

    train_dataset = Dataset(
        X_train,
        cat_features=["estacion"]
    )

    test_dataset = Dataset(
        X_test,
        cat_features=["estacion"]
    )

    check = DataDuplicates()

    print("\n--- Entrenamiento ---")
    train_result = check.run(train_dataset)
    print(train_result.value)

    print("\n--- Prueba ---")
    test_result = check.run(test_dataset)
    print(test_result.value)

    return train_result, test_result

def validate_data_drift(X_train, X_test):
    """Detecta cambios en las distribuciones usando Deepchecks."""

    print("\n===== DEEPCHECKS: DATA DRIFT =====")

    # Utilizamos muestras para reducir el tiempo de ejecución.
    sample_size = 10000

    train_sample = X_train.sample(
        n=min(sample_size, len(X_train)),
        random_state=42
    )

    test_sample = X_test.sample(
        n=min(sample_size, len(X_test)),
        random_state=42
    )

    # Convertimos los datos al formato de Deepchecks.
    train_dataset = Dataset(
        train_sample,
        cat_features=["estacion"]
    )

    test_dataset = Dataset(
        test_sample,
        cat_features=["estacion"]
    )

    # Creamos y ejecutamos la comprobación.
    check = FeatureDrift()

    result = check.run(
        train_dataset,
        test_dataset
    )

    print("\nRESULTADOS DE DATA DRIFT:")
    print(result.value)

    # Crear carpeta para los reportes.
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Guardar reporte HTML.
    report_path = reports_dir / "deepchecks_data_drift.html"

    # Eliminar el reporte anterior para evitar archivos duplicados.
    if report_path.exists():
        report_path.unlink()

    result.save_as_html(
        str(report_path),
        as_widget=False
    )

    print(f"\nReporte HTML generado en: {report_path}")

    return result

def generate_deepchecks_suite(X_train, X_test):
    """Genera un reporte integral de calidad de datos."""

    print("\n===== REPORTE INTEGRAL DE DEEPCHECKS =====")

    # Utilizamos muestras reproducibles.
    sample_size = 10000

    train_sample = X_train.sample(
        n=min(sample_size, len(X_train)),
        random_state=42
    )

    test_sample = X_test.sample(
        n=min(sample_size, len(X_test)),
        random_state=42
    )

    # Preparamos los conjuntos para Deepchecks.
    train_dataset = Dataset(
        train_sample,
        cat_features=["estacion"]
    )

    test_dataset = Dataset(
        test_sample,
        cat_features=["estacion"]
    )

    # Agrupamos las comprobaciones.
    suite = Suite(
        "MiBici - Validación de calidad de datos",
        PercentOfNulls(),
        DataDuplicates(),
        FeatureDrift()
    )

    # Ejecutamos la suite.
    result = suite.run(train_dataset, test_dataset)

    # Guardamos el reporte.
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "deepchecks_quality_report.html"

    # Eliminar el reporte anterior para evitar archivos duplicados.
    if report_path.exists():
        report_path.unlink()

    result.save_as_html(
        str(report_path),
        as_widget=False
    )

    print(f"\nReporte integral generado en: {report_path}")

    return result


if __name__ == "__main__":
    X_train, X_test = load_data()

    validate_data_types(X_train, X_test)
    validate_missing_values(X_train, X_test)
    validate_duplicates(X_train, X_test)

    validate_nulls_deepchecks(X_train, X_test)
    validate_duplicates_deepchecks(X_train, X_test)

    validate_data_drift(X_train, X_test)

    generate_deepchecks_suite(X_train, X_test)
