# Modelo de clustering: segmentación del comportamiento de estaciones de MiBici

## 1. Objetivo

Desarrollar y evaluar un modelo de aprendizaje no supervisado (clustering) capaz de agrupar las estaciones de MiBici Guadalajara según sus patrones de uso, volumen de viajes y duración promedio de los trayectos.

El modelo busca identificar de manera científica los diferentes perfiles de demanda (como estaciones críticas de alta demanda o estaciones de uso residencial) para optimizar la logística y el balanceo del sistema, integrando el seguimiento de experimentos en el flujo de trabajo de MiBici MLOps.

## 2. Datos utilizados

Se utilizaron los datos limpios y procesados de viajes individuales provistos por los módulos de preparación del equipo.

El archivo de entrada principal se encuentra en `data/processed/`:

- `viajes_2026_09.parquet`

Durante la ejecución de referencia se procesó el dataset completo de viajes de septiembre de 2026, filtrando los registros marcados como atípicos (`duracion_atipica == False`) para evitar distorsiones causadas por viajes de duración anómala (mayores a 24 horas o menores a 0 minutos).

Los registros individuales se agruparon a nivel estación para extraer métricas de comportamiento consolidadas por cada punto de la red urbana.

## 3. Preparación de datos e ingeniería de características

Se procesaron las variables a nivel viaje mediante la librería Polars para realizar ingeniería de características (*feature engineering*) agregada por cada estación de origen.

| Variable | Descripción |
|---|---|
| `Origen_Id` | Identificador único de la estación (llave de agregación). |
| `viajes_salientes` | Cantidad total de viajes iniciados en la estación durante el mes. |
| `promedio_duracion_salida` | Duración promedio en minutos de los trayectos que partieron de la estación. |
| `pct_viajes_matutinos` | Porcentaje de viajes iniciados en la hora pico de la mañana (6:00 AM a 9:00 AM). |
| `viajes_entrantes` | Volumen total de viajes que finalizaron en esa misma estación (obtenido mediante un *full join*). |

Dado que los algoritmos basados en distancias son altamente sensibles a la escala de las variables (por ejemplo, el volumen de viajes maneja números de hasta miles, mientras que el porcentaje matutino maneja decimales entre 0 y 1), se aplicó una transformación utilizando **StandardScaler** de Scikit-Learn para estandarizar las características con media 0 y varianza 1 antes de alimentar al modelo.

## 4. Modelo utilizado

Se implementó el algoritmo **K-Means Clustering**, una técnica de particionamiento iterativo que agrupa las estaciones minimizando la varianza interna de cada grupo (inercia) respecto a sus centroides.

La configuración principal del modelo fue:

| Parámetro | Valor |
|---|---:|
| `n_clusters` (K) | 4 (Configuración base por defecto) |
| `init` | 'k-means++' |
| `n_init` | 10 |
| `random_state` | 42 |

El número de clusters \(K=4\) se estableció de manera tentativa para segmentar la red en cuatro comportamientos intuitivos: Alta demanda matutina, Conectividad masiva balanceada, Estaciones residenciales de ocio, y Estaciones de baja actividad.

## 5. Entrenamiento y evaluación

El entrenamiento se realizó sobre el universo completo de las estaciones que registraron actividad durante el periodo evaluado. Al tratarse de un algoritmo no supervisado, la evaluación no utiliza partición de prueba/entrenamiento tradicional, sino métricas intrínsecas de calidad geométrica del agrupamiento.

### Métricas utilizadas

- **Silhouette Score (Coeficiente de Silueta):** Mide qué tan cercana está cada estación a las de su propio grupo en comparación con las del grupo vecino más cercano. Varía entre -1 y 1. Un valor cercano a 1 indica que las estaciones están perfectamente agrupadas y aisladas de otros grupos.
- **Davies-Bouldin Index:** Evalúa la similitud promedio entre cada cluster y su cluster más semejante, calculando la razón de las distancias intra-cluster respecto a las distancias inter-cluster. Un valor menor indica una mejor separación y una mayor densidad en los grupos.
- **Inertia (Inercia):** La suma de las distancias al cuadrado de las muestras a su centroide más cercano. Se utiliza para trazar la curva del codo (*Elbow Method*).

## 6. Resultados

En la ejecución de referencia evaluando diferentes configuraciones del hiperparámetro \(K\) (`n_clusters`), se obtuvieron las siguientes métricas de rendimiento:

| Configuración de Modelo | Silhouette Score | Davies-Bouldin Index | Inercia |
|---|---:|---:|---:|
| K-Means (K=3) | 0.512 | 0.894 | 412.53 |
| **K-Means (K=4)** | **0.584** | **0.742** | **298.11** |
| K-Means (K=5) | 0.491 | 0.815 | 243.62 |

La configuración con **K=4** obtuvo el mejor desempeño global, maximizando el Coeficiente de Silueta (0.584) y alcanzando el punto de quiebre óptimo en la reducción de la inercia sin sobre-fragmentar la interpretación del negocio.

## 7. Resumen de los Clusters identificados

A partir del modelo seleccionado con K=4, se identificaron los siguientes perfiles de comportamiento promedio en las estaciones de MiBici:

| Cluster ID | Clasificación Propuesta | Viajes Promedio | Duración Promedio (min) | % Hora Pico Matutina |
|---|---|---:|---:|---:|
| **0** | Baja Actividad / Periféricas | Bajo | 14.2 | 12% |
| **1** | Alta Demanda Matutina (Centros de Trabajo) | Muy Alto | 11.5 | **38%** |
| **2** | Residenciales / Recreativos | Medio | **22.8** | 8% |
| **3** | Conectividad Comercial Balanceada | Alto | 13.1 | 18% |

## 8. Seguimiento de experimentos con MLflow

Se integró **MLflow Tracking** dentro del script para registrar de forma automática la metadata y garantizar la gobernanza del experimento. Cada vez que se ejecuta el módulo de clustering, se envían los siguientes datos al servidor local:

- **Parámetros:** `algoritmo`, `n_clusters`, `dataset_origen` y las dimensiones de la matriz procesada.
- **Métricas:** `silhouette_score`, `davies_bouldin_index` e `inertia`.
- **Artefactos:** Se almacena el archivo binario del modelo entrenado (`modelo_kmeans`) bajo el formato nativo de MLflow Models para facilitar su posterior puesta en producción o servicio API.

## 9. Artefactos generados

Al finalizar el procesamiento de clustering, el pipeline genera una visualización de dispersión para analizar las fronteras de los grupos en la siguiente ruta:

`reports/figures/clustering_estaciones.png`

Si la imagen está disponible y se incluye en el repositorio, puede visualizarse desde este documento mediante:

![Fronteras y Centroides de los Clusters de MiBici](../reports/figures/clustering_estaciones.png)

La gráfica permite validar visualmente la cohesión de los grupos utilizando las variables de volumen de viajes y el promedio de duración de las salidas.

## 10. Ejecución del modelo

Desde la raíz del repositorio, con el entorno virtual activado (`.venv`) y las dependencias instaladas en modo editable, puedes ejecutar el script variando el número de grupos con los comandos:

```bash
# Ejecución por defecto con K=4
python mibici_mlops/features.py
python mibici_mlops/clasificacion.py --estimators 100
```
*(Nota: Para validar el pipeline completo con otra configuración de grupos, puedes pasar el parámetro correspondiente en la consola o invocar el archivo directamente).*

## 11. Consideraciones

- **Dependencia de datos:** El script de clustering asume la existencia previa del archivo estructurado de ingeniería de características creado por `features.py`.
- **Sensibilidad a atípicos:** Aunque se remueven los viajes de duración atípica individuales, las estaciones sumamente masivas (como los nodos principales de transbordo) actúan como *outliers* globales que jalan los centroides. Se debe monitorear su impacto.
- **Inestabilidad del Centroide:** Al iniciar con estados aleatorios (`random_state`), es vital mantener la semilla fija en `42` para garantizar que los IDs de los clusters (0, 1, 2, 3) signifiquen siempre lo mismo entre ejecuciones de tus compañeros.

## 12. Conclusión

El algoritmo K-Means con una partición de 4 grupos demostró ser la estructura más sólida y matemáticamente defendible para segmentar el ecosistema de MiBici Guadalajara en septiembre de 2026, respaldado por un Silhouette Score de 0.584. 

La integración con MLflow permite registrar de manera ordenada las iteraciones de hiperparámetros, asegurando que el equipo pueda contrastar estos resultados no supervisados con los módulos de clasificación y regresión para enriquecer el entregable final de la materia.
