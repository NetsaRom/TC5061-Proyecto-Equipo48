# MiBici MLOps

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto MLOps para analizar viajes de MiBici Guadalajara mediante modelos de regresión, clasificación y clustering, con pipelines reproducibles y seguimiento de experimentos en MLflow.

## Instalación del proyecto

Este proyecto utiliza Python 3.11 y sigue la estructura de Cookiecutter Data Science.

### 1. Clonar el repositorio

```bash
git clone https://github.com/NetsaRom/TC5061-Proyecto-Equipo48.git
cd TC5061-Proyecto-Equipo48
```

### 2. Crear un entorno virtual

```bash
python3.11 -m venv .venv
```

Activar el entorno virtual:

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

El archivo `requirements.txt` contiene versiones fijas de las dependencias principales utilizadas para procesamiento de datos, machine learning, MLflow y pruebas.

### 4. Verificar la instalación

```bash
python -m pip check
python -m pytest tests/test_data.py -v
```

Estas instrucciones permiten preparar el entorno de desarrollo y ejecutar las pruebas de ingesta y trazabilidad del proyecto.

## Ingesta y procesamiento de datos

El proyecto utiliza datos abiertos del sistema de bicicletas públicas MiBici de Guadalajara, Jalisco, México.

Para esta etapa se seleccionó el periodo comprendido entre **julio y septiembre de 2026**.

### Ejecutar el pipeline

Desde la raíz del repositorio, con el entorno virtual activado, ejecutar:

```bash
python -m mibici_mlops.dataset procesar-periodos --inicio 2026-07 --fin 2026-09
```

Este proceso realiza las siguientes operaciones:

1. Obtiene los archivos CSV mensuales de viajes desde la fuente de datos abiertos de MiBici.
2. Lee y limpia los registros de viajes.
3. Genera archivos Parquet para su posterior análisis y modelado.
4. Descarga y procesa el catálogo de estaciones, cuando está habilitado en la configuración.
5. Genera el archivo `data/manifest.json` con metadatos de los archivos procesados.

### Resultados del procesamiento

En la ejecución de referencia se obtuvieron:

| Periodo | Viajes procesados |
|---|---:|
| Julio de 2026 | 388,257 |
| Agosto de 2026 | 393,524 |
| Septiembre de 2026 | 489,298 |
| **Total** | **1,271,079** |

También se procesó el catálogo de septiembre de 2026, con **484 estaciones**.

Los archivos resultantes se almacenan principalmente en `data/raw/`, `data/external/` y `data/processed/`.

### Manifiesto de trazabilidad

El archivo `data/manifest.json` registra información de los archivos utilizados y generados durante la ingesta:

- Nombre y ruta relativa del archivo.
- Tamaño en bytes.
- Número de registros, cuando corresponde a un archivo Parquet.
- Hash SHA-256 para comprobar su integridad.
- Fecha de generación del manifiesto en UTC.

El manifiesto se versiona en Git, mientras que los archivos CSV y Parquet permanecen excluidos mediante `.gitignore`.

**Nota de reproducibilidad:** El manifiesto permite identificar diferencias entre versiones de los archivos. Si los datos publicados por la fuente cambian, una nueva descarga puede producir hashes distintos. Para reproducir exactamente una versión anterior es necesario disponer de los mismos archivos originales.

### Configuración

Los valores predeterminados del periodo y las rutas del proyecto se centralizan en `mibici_mlops/config.py`.

Los argumentos `--inicio` y `--fin` permiten seleccionar otro intervalo mensual sin modificar el código fuente.

## Modelos de Machine Learning

El proyecto contempla el desarrollo y evaluación de modelos de Machine Learning para analizar los viajes de MiBici Guadalajara mediante tres enfoques principales: **regresión, clasificación y clustering**. Estos modelos permiten abordar distintos problemas de análisis y predicción a partir de los datos del sistema de bicicletas públicas.

El código fuente de los modelos, incluyendo sus implementaciones y componentes relacionados, se encuentra en el directorio `mibici_mlops/modeling/`.

### Modelos considerados

- **Regresión:** predicción de la cantidad de viajes por estación y hora mediante un modelo Random Forest Regressor.
- **Clasificación:** desarrollo de modelos para asignar categorías a las observaciones según el objetivo definido.
- **Clustering:** identificación de grupos de observaciones con características similares mediante técnicas de aprendizaje no supervisado.

Para consultar los detalles técnicos de cada enfoque, incluyendo objetivos, preparación de datos, metodología, configuración y resultados de evaluación, revisa la documentación disponible en `docs/README.md`.

## MLflow: seguimiento de experimentos y pipelines reproducibles

El proyecto incorpora **MLflow Projects** para ejecutar los flujos de ingesta y entrenamiento de forma estandarizada, y **MLflow Tracking** para registrar los parámetros, las métricas y los artefactos generados por cada ejecución.

La configuración de los puntos de entrada está definida en `MLproject`. El entorno de ejecución se describe en `python_env.yaml`, y `mibici_mlops/modeling/train_project.py` selecciona el flujo de entrenamiento a partir del parámetro `question`.

### Requisitos

- Python 3.11.
- Dependencias instaladas desde `requirements.txt`.
- MLflow disponible en el entorno virtual.
- Acceso a las fuentes de datos para ejecutar la ingesta o disponibilidad de los datos procesados para entrenar.

### Puntos de entrada

| Punto de entrada | Descripción |
|---|---|
| `ingest` | Ejecuta la ingesta y la generación de los datos procesados y características. |
| `train` | Ejecuta uno de los flujos de modelado disponibles según el parámetro `question`. |
| `experiment_p1` | Ejecuta la búsqueda de hiperparámetros y la evaluación final de P1 (regresión), registrando corridas anidadas, métricas, modelos y artefactos en MLflow. |

El punto de entrada `train` admite los siguientes parámetros:

| Parámetro | Descripción | Valor predeterminado |
|---|---|---|
| `question` | Flujo de modelado: `p1`, `p2` o `p3`. Es obligatorio. | Sin valor predeterminado |
| `model` | Identificador del modelo solicitado por el flujo. | `default` |
| `n_estimators` | Número de estimadores para los modelos que utilizan este parámetro. | `100` |
| `max_depth` | Profundidad máxima para los modelos que utilizan este parámetro. | `15` |
| `n_clusters` | Número de clusters para el flujo que utiliza clustering. | `4` |
| `test_size` | Proporción de datos reservada para prueba en los flujos que utilizan este parámetro. | `0.2` |

No todos los parámetros aplican a todos los flujos. La configuración efectiva depende de la implementación de cada modelo.

### Configurar un servidor local de tracking

Para iniciar una instancia local de MLflow Tracking desde la raíz del repositorio:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow-tracking.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Este comando mantiene el servidor accesible únicamente desde la máquina local. La base de datos de tracking y los artefactos se guardan localmente; no constituyen por sí mismos una configuración compartida para todo el equipo.

En otra terminal, activa el entorno virtual y configura el URI de tracking:

**macOS / Linux:**

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

**Windows (PowerShell):**

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

La interfaz web estará disponible en:

```text
http://127.0.0.1:5000
```

Para un entorno colaborativo, el equipo debe acordar un servidor de tracking accesible para sus integrantes y un almacenamiento de artefactos compartido. No se deben subir a Git la base de datos de tracking ni los directorios de artefactos.

### Ejecutar los pipelines con MLflow Projects

Ejecuta los comandos desde la raíz del repositorio, con el entorno virtual activado y `MLFLOW_TRACKING_URI` configurado según el entorno que utilices.

**Ingesta y generación de características:**

```bash
mlflow run . -e ingest --env-manager local
```

**Entrenamiento del flujo P1 (regresión):**

```bash
mlflow run . -e train -P question=p1 --experiment-name mibici-equipo48 --env-manager local
```

**Experimentación P1 — búsqueda de hiperparámetros y evaluación final:**

```bash
mlflow run . -e experiment_p1 --experiment-name mibici-equipo48 --env-manager local
```

Este punto de entrada ejecuta `mibici_mlops/modeling/experiment_p1.py`, que realiza una búsqueda de tres configuraciones de Random Forest mediante validación temporal, registra corridas anidadas en MLflow y compara los candidatos contra el modelo original utilizando un conjunto de prueba reservado.

La configuración seleccionada utiliza 300 árboles y profundidad máxima de 20. En la ejecución de referencia obtuvo un MAE de validación de 0.806701 y un MAE de prueba de 0.970725.

**Nota:** este comando vuelve a ejecutar todos los entrenamientos de P1 y puede tardar considerablemente. Requiere disponer de los archivos Parquet procesados correspondientes a julio, agosto y septiembre de 2026. Para consultar los resultados existentes no es necesario ejecutarlo nuevamente.

**Entrenamiento del flujo P2 (clasificación):**

```bash
mlflow run . -e train -P question=p2 --experiment-name mibici-equipo48 --env-manager local
```

**Entrenamiento del flujo P3 (clustering):**

```bash
mlflow run . -e train -P question=p3 --experiment-name mibici-equipo48 --env-manager local
```

Se utiliza `--env-manager local` para ejecutar los comandos con el entorno virtual actual, cuyas dependencias deben estar instaladas. Si se prefiere que MLflow cree y gestione un entorno aislado, se puede utilizar otro gestor compatible con la versión instalada de MLflow y configurarlo de acuerdo con la documentación de esa versión.

### Parámetros personalizados

Los parámetros del punto de entrada `train` se pueden sobrescribir con `-P`. Por ejemplo:

```bash
mlflow run . -e train \
  -P question=p1 \
  -P n_estimators=150 \
  -P max_depth=12 \
  -P test_size=0.25 \
  --experiment-name mibici-equipo48 \
  --env-manager local
```

Los parámetros que no utilice el flujo seleccionado no cambian necesariamente su comportamiento; consulta el código de cada modelo para conocer cuáles tienen efecto.

### Parámetros, métricas y artefactos registrados

Las ejecuciones registran información para facilitar la trazabilidad y la comparación de experimentos. Según el flujo, puede incluir:

- Parámetros de ejecución y configuración del modelo.
- Métricas de evaluación.
- Artefactos de evaluación y visualización.
- `data_lineage/manifest.json`, con información de trazabilidad de los datos usados.
- El modelo serializado en MLflow, cuando la ejecución lo registra.

Entre los artefactos específicos de los flujos se encuentran:

| Flujo | Métricas o artefactos |
|---|---|
| P1 — Regresión | MAE, RMSE y visualización de resultados en `p1_regression.png`. |
| P2 — Clasificación | Accuracy, F1, ROC-AUC y reporte de clasificación en `evaluacion/p2_classification_report.txt`. |
| P3 — Clustering | Inercia, silhouette, Davies-Bouldin y visualización en `p3_clustering.png`. |

Los nombres y las métricas disponibles dependen de la implementación y de que la ejecución termine correctamente.

### Consultar resultados

Después de ejecutar un pipeline, abre la interfaz de MLflow en `http://127.0.0.1:5000` si estás usando el servidor local descrito arriba. Selecciona el experimento correspondiente para comparar ejecuciones, revisar parámetros y métricas, y consultar los artefactos registrados.

Una ejecución fallida puede aparecer en el experimento. Para interpretar resultados, distingue las ejecuciones con estado `FINISHED` de aquellas con estado `FAILED`; no tomes una ejecución fallida como evidencia de un entrenamiento completado.

### Estado y alcance del registro de modelos

El registro de un modelo y la disponibilidad de un artefacto no implican que este se encuentre desplegado en producción. Antes de promover un modelo para uso operativo, se deben revisar sus métricas, la idoneidad del objetivo, la trazabilidad de los datos y los requisitos de despliegue y monitoreo.

En particular, la etiqueta de alta demanda utilizada en el flujo P2 es un objetivo definido por el proyecto y debe interpretarse de acuerdo con su construcción. No debe considerarse automáticamente una medición directa de desabasto ni una garantía de demanda futura.

#### Modelo registrado P1 — Predicción de demanda

El modelo seleccionado para P1 se registró en MLflow Model Registry con la siguiente configuración:

| Propiedad | Valor |
|---|---|
| Nombre registrado | `mibici-equipo48-p1-demanda` |
| Versión | 1 |
| Alias de producción | `champion` |
| Algoritmo | RandomForestRegressor |
| Número de árboles | 300 |
| Profundidad máxima | 20 |
| Experimento de origen | `mibici-equipo48` |
| Corrida de origen | `P1_candidate_3` |
| Commit de entrenamiento | `4cec099` |
| Datos utilizados | MiBici Guadalajara, julio–septiembre de 2026 |

**Esquema de ciclo de vida:** se utilizan aliases de MLflow Model Registry en lugar de las etapas clásicas (*Staging*, *Production* y *Archived*). El alias `champion` identifica la versión seleccionada para uso en producción y permite actualizarla sin modificar las referencias utilizadas por los consumidores del modelo.

La trazabilidad de los datos se conserva mediante el archivo `data/manifest.json` asociado a la corrida, que incluye los hashes SHA-256 de los archivos procesados.

**Cargar el modelo seleccionado:**

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.pyfunc.load_model(
    "models:/mibici-equipo48-p1-demanda@champion"
)

print("Modelo champion cargado correctamente")
```

La carga mediante nombre y alias fue verificada exitosamente en el entorno local. La asignación del alias representa la selección de la versión de producción dentro del registro; no implica que exista un servicio de inferencia desplegado.

Para reproducir el acceso al modelo registrado se requiere utilizar el mismo backend de MLflow Registry o disponer de una copia de sus metadatos y artefactos.

## Project Organization

```text
├── LICENSE
├── Makefile
├── README.md
├── MLproject                  <- Entry points de MLflow Projects
├── python_env.yaml            <- Entorno Python declarado para MLflow Projects
├── data
│   ├── external               <- Datos de fuentes externas, como el catálogo
│   ├── interim                <- Datos intermedios transformados
│   ├── processed              <- Datasets preparados para modelado
│   └── raw                    <- Archivos originales descargados
├── docs                       <- Documentación técnica del proyecto
├── models                     <- Modelos y resultados, si se guardan localmente
├── notebooks                  <- Notebooks de exploración y análisis
├── pyproject.toml             <- Metadatos y configuración del proyecto
├── references                 <- Diccionarios de datos y material de referencia
├── reports                    <- Reportes y resultados de análisis
│   └── figures                <- Figuras generadas para reportes
├── requirements.txt           <- Dependencias del entorno
├── setup.cfg                  <- Configuración de herramientas
└── mibici_mlops
    ├── __init__.py
    ├── config.py              <- Variables y configuración
    ├── dataset.py             <- Descarga y procesamiento de datos
    ├── features.py            <- Generación de características
    ├── project.py             <- Orquestación de ingesta para MLflow Projects
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py         <- Inferencia con modelos entrenados
    │   ├── train.py           <- Entrenamiento de modelos
    │   └── train_project.py   <- Punto de entrada de entrenamiento de MLflow Projects
    └── plots.py               <- Visualizaciones
```

## Buenas prácticas de reproducibilidad

- Mantener las versiones de dependencias sincronizadas con `requirements.txt`.
- Revisar los parámetros, métricas y artefactos de cada ejecución antes de comparar modelos.
- Conservar los manifiestos de trazabilidad para identificar la versión de los datos usada.
- No versionar bases de datos de tracking ni artefactos generados localmente.
- Documentar cualquier cambio en la fuente de datos, el periodo de análisis o la definición de las variables objetivo.

**Nota sobre los conteos de datos:** Los resultados de referencia de la ingesta y los conteos leídos por cada flujo de modelado pueden diferir por filtros, registros válidos, transformaciones o por el conjunto de estaciones representado en las características. Antes de comparar esos conteos, verifica el manifiesto de datos y la lógica de preparación utilizada por cada flujo.
