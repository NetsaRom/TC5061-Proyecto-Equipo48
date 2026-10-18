# P2. Modelo de clasificación: demanda de estaciones MiBici

## 1. Objetivo

Desarrollar un modelo de clasificación para identificar estaciones de MiBici Guadalajara con diferentes niveles de demanda, utilizando información histórica de viajes.

Este modelo forma parte del proyecto MiBici MLOps y busca explorar cómo las características de los viajes pueden ayudar a distinguir estaciones con alta y baja demanda.

La pregunta planteada para P2 es:

**¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro?**

La implementación actual utiliza la variable `alta_demanda` como objetivo exploratorio. Esta etiqueta representa demanda relativa y todavía no equivale directamente al riesgo futuro de disponibilidad.

## 2. Datos utilizados

Se utilizaron los archivos Parquet correspondientes a julio, agosto y septiembre de 2026:

- `data/processed/viajes_2026_07.parquet`
- `data/processed/viajes_2026_08.parquet`
- `data/processed/viajes_2026_09.parquet`

Los datos contienen información sobre el inicio y fin de los viajes, las estaciones de origen y destino y la duración de cada recorrido.

Las columnas principales utilizadas son:

| Columna | Descripción |
|---|---|
| `Inicio_del_viaje` | Fecha y hora de inicio del viaje. |
| `Fin_del_viaje` | Fecha y hora de finalización del viaje. |
| `Origen_Id` | Identificador de la estación de origen. |
| `Destino_Id` | Identificador de la estación de destino. |
| `duracion_min` | Duración del viaje en minutos. |

## 3. Generación de características para clasificación

Antes de entrenar el modelo, se implementó un proceso de ingeniería de características para transformar los registros de viajes en un conjunto de datos agrupado por estación.

### 3.1. Ejecución del proceso

Desde la raíz del repositorio, con el entorno virtual activado, se ejecutó:

```powershell
python -m mibici_mlops.modeling.features_clasificacion
```

Este proceso realiza las siguientes operaciones:

1. Carga los archivos Parquet de julio, agosto y septiembre de 2026.
2. Extrae la hora de inicio de cada viaje.
3. Agrupa los viajes por estación de origen para calcular características de las salidas.
4. Calcula la duración promedio de los viajes de salida.
5. Calcula el porcentaje de viajes matutinos, considerando el intervalo de las 6:00 a las 11:59.
6. Agrupa los viajes por estación de destino para calcular la cantidad de viajes entrantes.
7. Combina las características de salidas y entradas por estación.
8. Calcula la mediana de los viajes salientes para definir un umbral de demanda relativa.
9. Genera la variable objetivo `alta_demanda`, codificada como `0` o `1`.
10. Guarda el conjunto de características generado en formato Parquet.

### 3.2. Características generadas

| Variable | Descripción |
|---|---|
| `estacion` | Identificador de la estación. |
| `viajes_salientes` | Cantidad de viajes que salen de la estación. |
| `promedio_duracion_salida` | Duración promedio de los viajes de salida. |
| `pct_viajes_matutinos` | Porcentaje de salidas registradas durante la mañana. |
| `viajes_entrantes` | Cantidad de viajes que llegan a la estación. |
| `alta_demanda` | Etiqueta binaria de demanda relativa. |

### 3.3. Definición de la variable objetivo

La variable `alta_demanda` se construye comparando la cantidad de viajes salientes de cada estación con la mediana de los viajes salientes del conjunto.

- `0`: cantidad de viajes salientes inferior a la mediana.
- `1`: cantidad de viajes salientes igual o superior a la mediana.

Esta definición permite explorar la clasificación de estaciones según su demanda relativa. No representa directamente el riesgo de quedarse sin bicicletas o sin anclajes disponibles.

### 3.4. Archivo generado

El conjunto de características se almacena en:

`data/processed/features_estaciones.parquet`

Este archivo funciona como entrada para el entrenamiento del modelo de clasificación.

## 4. Algoritmo utilizado

Se implementó **Random Forest Classifier**, un algoritmo de aprendizaje supervisado que combina múltiples árboles de decisión para clasificar observaciones.

La configuración utilizada en la ejecución documentada fue:

| Parámetro | Valor |
|---|---:|
| `n_estimators` | 200 |
| `random_state` | 42 |
| `test_size` | 0.20 |
| `class_weight` | `balanced` |
| `n_jobs` | -1 |

Los datos se dividieron en conjuntos de entrenamiento y prueba mediante una división estratificada, conservando la proporción de las clases.

## 5. Entrenamiento del modelo

Una vez generado `features_estaciones.parquet`, se ejecutó el entrenamiento mediante el siguiente comando:

```powershell
python -m mibici_mlops.modeling.train_p2_classification --estimators 200
```

El script realiza las siguientes operaciones:

1. Carga el conjunto de características procesadas.
2. Verifica la existencia de las columnas requeridas.
3. Elimina registros con valores nulos en las variables seleccionadas.
4. Separa las características predictoras de la variable objetivo.
5. Divide los datos en entrenamiento y prueba.
6. Entrena el clasificador Random Forest.
7. Genera predicciones y probabilidades para las observaciones de prueba.
8. Calcula las métricas de evaluación.
9. Registra los parámetros, las métricas y el modelo en MLflow.

Las características utilizadas para el entrenamiento son:

- `promedio_duracion_salida`
- `pct_viajes_matutinos`
- `viajes_entrantes`

La variable objetivo es `alta_demanda`.

## 6. Métricas de evaluación

Para evaluar el desempeño del clasificador se utilizaron las siguientes métricas:

- **Accuracy:** proporción de predicciones correctas sobre el total de observaciones evaluadas.
- **F1-score:** media armónica entre precisión y recall.
- **ROC-AUC:** capacidad del modelo para distinguir entre las clases a partir de las probabilidades predichas.

También se generó una matriz de confusión para analizar los aciertos y errores de clasificación.

## 7. Resultados obtenidos

En la ejecución evaluada con 200 árboles se obtuvieron los siguientes resultados:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9681 |
| F1-score | 0.9684 |
| ROC-AUC | 0.9968 |
| Observaciones de prueba | 94 |

### 7.1. Reporte de clasificación

| Clase | Precisión | Recall | F1-score | Observaciones |
|---|---:|---:|---:|---:|
| 0 | 0.98 | 0.96 | 0.97 | 47 |
| 1 | 0.96 | 0.98 | 0.97 | 47 |

### 7.2. Matriz de confusión

| | Predicción: clase 0 | Predicción: clase 1 |
|---|---:|---:|
| Clase real 0 | 45 | 2 |
| Clase real 1 | 1 | 46 |

El modelo clasificó correctamente 91 de las 94 observaciones de prueba y se equivocó en 3.

Estos resultados muestran un buen desempeño para distinguir las clases definidas en este experimento. Sin embargo, deben interpretarse considerando el tamaño del conjunto de prueba y la definición de la variable objetivo.

## 8. Seguimiento de experimentos con MLflow

Se integró MLflow para registrar los experimentos de clasificación y facilitar el seguimiento de los resultados.

El experimento se identifica con el nombre:

`MiBici_Clasificacion_Estaciones`

Se registran los parámetros del modelo, las métricas de evaluación y el modelo entrenado.

El almacenamiento de los experimentos utiliza una base de datos SQLite configurada en `mlflow.db`. Para consultar los experimentos desde la raíz del proyecto, se puede ejecutar:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Si la base de datos se encuentra en una ruta absoluta diferente, debe utilizarse esa misma ubicación al iniciar la interfaz.

## 9. Archivos relacionados

Los principales archivos involucrados en el flujo de clasificación son:

| Archivo | Función |
|---|---|
| `mibici_mlops/modeling/features_clasificacion.py` | Generación de características por estación. |
| `mibici_mlops/modeling/train_p2_classification.py` | Entrenamiento y evaluación del clasificador. |
| `data/processed/viajes_2026_07.parquet` | Viajes de julio de 2026. |
| `data/processed/viajes_2026_08.parquet` | Viajes de agosto de 2026. |
| `data/processed/viajes_2026_09.parquet` | Viajes de septiembre de 2026. |
| `data/processed/features_estaciones.parquet` | Características utilizadas para el entrenamiento. |
| `mlflow.db` | Base de datos de seguimiento de experimentos. |

## 10. Consideraciones y trabajo futuro

La implementación actual clasifica estaciones según su demanda relativa, utilizando la mediana de los viajes salientes como umbral.

Para responder directamente a la pregunta P2, será necesario definir una etiqueta que represente el riesgo futuro de quedarse sin bicicletas o sin anclajes. Esto podría requerir registros de disponibilidad de las estaciones o una definición proxy debidamente justificada.

También será importante evaluar el modelo utilizando periodos posteriores a los empleados para construir las características, evitando que información futura influya en las predicciones.

## 11. Conclusión

Se implementó un flujo de clasificación que transforma los registros de viajes en características por estación, entrena un modelo Random Forest y registra sus resultados en MLflow.

En la ejecución documentada, el clasificador obtuvo un Accuracy de 96.81 %, un F1-score de 96.84 % y un ROC-AUC de 99.68 %.

Estos resultados corresponden a la clasificación de demanda relativa. Para considerar el modelo una solución al problema de riesgo futuro de disponibilidad, será necesario desarrollar y validar una variable objetivo que represente dicho riesgo.