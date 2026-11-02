
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from mibici_mlops.modeling.train_p1_regression import (
    build_model,
    evaluate_model,
)


def crear_datos_entrenamiento():
    """Genera datos pequeños para probar el entrenamiento."""

    X = pd.DataFrame({
        "estacion": [
            "A", "A", "A", "A", "A", "A",
            "B", "B", "B", "B", "B", "B",
        ],
        "hora_dia": [6, 7, 8, 9, 10, 11] * 2,
        "dia_semana": [0, 1, 2, 3, 4, 5] * 2,
        "mes": [9] * 12,
        "fin_semana": [0, 0, 0, 0, 0, 1] * 2,
        "viajes_lag_1": [2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6],
        "viajes_lag_24": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
        "viajes_lag_168": [2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3],
        "promedio_24h": [2, 3, 3, 4, 5, 6, 1, 2, 2, 3, 4, 5],
    })

    y = pd.Series(
        [3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7],
        name="target_viajes_siguiente_hora"
    )

    return X, y


def test_build_model():
    """Verifica que el modelo se construya correctamente."""

    X, _ = crear_datos_entrenamiento()

    model = build_model(
        features=list(X.columns),
        n_estimators=10,
        max_depth=5,
        random_state=42,
        n_jobs=1
    )

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "regressor" in model.named_steps
    assert model.named_steps["regressor"].n_estimators == 10
    assert model.named_steps["regressor"].random_state == 42


def test_evaluate_model():
    """Verifica el entrenamiento, las predicciones y las métricas."""

    X, y = crear_datos_entrenamiento()

    X_train = X.iloc[:8]
    y_train = y.iloc[:8]

    X_test = X.iloc[8:]
    y_test = y.iloc[8:]

    model = build_model(
        features=list(X.columns),
        n_estimators=10,
        max_depth=5,
        random_state=42,
        n_jobs=1
    )

    metrics, predictions = evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    assert len(predictions) == len(y_test)
    assert "test_mae" in metrics
    assert "test_rmse" in metrics
    assert metrics["test_mae"] >= 0
    assert metrics["test_rmse"] >= 0
    assert np.isfinite(predictions).all()


def test_training_reproducibility():
    """Verifica que dos entrenamientos con la misma semilla
    produzcan las mismas predicciones.
    """

    X, y = crear_datos_entrenamiento()

    X_train = X.iloc[:8]
    y_train = y.iloc[:8]

    X_test = X.iloc[8:]
    y_test = y.iloc[8:]

    model_1 = build_model(
        features=list(X.columns),
        n_estimators=20,
        max_depth=5,
        random_state=42,
        n_jobs=1
    )

    model_2 = build_model(
        features=list(X.columns),
        n_estimators=20,
        max_depth=5,
        random_state=42,
        n_jobs=1
    )

    _, predictions_1 = evaluate_model(
        model_1,
        X_train,
        y_train,
        X_test,
        y_test
    )

    _, predictions_2 = evaluate_model(
        model_2,
        X_train,
        y_train,
        X_test,
        y_test
    )

    np.testing.assert_array_equal(
        predictions_1,
        predictions_2
    )
