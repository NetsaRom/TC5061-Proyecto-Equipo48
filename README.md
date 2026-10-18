# MiBici MLOps

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto MLOps para analizar viajes de MiBici Guadalajara mediante modelos de regresion, clasificacion y clustering, con pipelines reproducibles y seguimiento de experimentos en MLflow.

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

## Modelos de Machine Learning

El proyecto contempla el desarrollo y evaluación de modelos de Machine Learning para analizar la demanda de viajes de MiBici Guadalajara, incluyendo tareas de regresión, clasificación y clustering. Se busca generar modelos reproducibles y dar seguimiento a sus experimentos mediante MLflow.

## Modelos de Machine Learning

El proyecto contempla el desarrollo y evaluación de modelos de Machine Learning para analizar los viajes de MiBici Guadalajara mediante tres enfoques principales: **regresión, clasificación y clustering**. Estos modelos permiten abordar distintos problemas de análisis y predicción a partir de los datos del sistema de bicicletas públicas.

La documentación técnica de los modelos, incluyendo sus objetivos, preparación de datos, metodología, configuración y resultados de evaluación, se encuentra en el siguiente documento:

Consulta la [documentación de los modelos de Machine Learning](docs/README.md) para conocer los detalles de cada enfoque.

### Modelos considerados

- **Regresión:** predicción de la cantidad de viajes por estación y hora mediante un modelo Random Forest Regressor.
- **Clasificación:** desarrollo de modelos para asignar categorías a las observaciones según el objetivo definido.
- **Clustering:** identificación de grupos de observaciones con características similares mediante técnicas de aprendizaje no supervisado.

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

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici_mlops and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici_mlops   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici_mlops a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

