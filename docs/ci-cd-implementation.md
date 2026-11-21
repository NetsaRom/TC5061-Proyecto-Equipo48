\# Implementación de Integración Continua (CI) – MiBici MLOps



\## 1. Objetivo



Implementar un flujo de Integración Continua (CI) mediante GitHub Actions para automatizar la validación del código, la ejecución de pruebas automatizadas y la construcción e integración de los servicios Docker del proyecto MiBici MLOps.



El objetivo es detectar errores de manera temprana y garantizar que los cambios incorporados al repositorio mantengan la calidad y funcionalidad del sistema.



\## 2. Herramientas utilizadas



| Herramienta | Función |

|---|---|

| GitHub Actions | Automatización del flujo de integración continua. |

| Python 3.11 | Entorno de ejecución del proyecto. |

| Ruff | Análisis estático y revisión de calidad del código. |

| Pytest | Ejecución de pruebas automatizadas. |

| Docker | Construcción y ejecución de contenedores. |

| Docker Compose | Orquestación de los servicios del proyecto. |

| FastAPI | Servicio de predicción mediante API. |

| MLflow | Gestión y servicio del modelo de aprendizaje automático. |



\## 3. Configuración de GitHub Actions



Se creó el archivo:



`.github/workflows/ci.yml`



Este archivo define el flujo de trabajo automatizado que se ejecuta mediante los siguientes eventos:



\- \*\*Push:\*\* ejecuta el pipeline cuando se suben cambios a cualquier rama.

\- \*\*Pull Request:\*\* ejecuta las validaciones cuando se solicita integrar cambios hacia las ramas `main` o `master`.

\- \*\*Workflow Dispatch:\*\* permite ejecutar manualmente el flujo desde GitHub Actions.



\## 4. Estructura del pipeline



El pipeline está compuesto por dos trabajos principales.



\### 4.1. Trabajo 1: Code Quality and Tests



Este trabajo valida la calidad del código y ejecuta las pruebas automatizadas.



Actividades realizadas:



1\. Descarga del repositorio mediante GitHub Actions.

2\. Configuración de Python 3.11.

3\. Instalación de las dependencias del proyecto, API y modelo P2.

4\. Ejecución de Ruff mediante:



&#x20;  `ruff check .`



5\. Ejecución de las pruebas automatizadas mediante:



&#x20;  `pytest tests/ -v --ignore=tests/test\_e2e.py`



Se excluyen las pruebas E2E porque requieren que los servicios externos de la aplicación estén levantados. Estas se ejecutan en el segundo trabajo.



\### 4.2. Trabajo 2: Docker Build and E2E Tests



Este trabajo depende de que el primero finalice correctamente.



Actividades realizadas:



1\. Descarga del repositorio.

2\. Verificación de Docker Compose.

3\. Construcción e inicio de los servicios mediante:



&#x20;  `docker compose up --build -d --wait --wait-timeout 180`



4\. Verificación del estado de los contenedores.

5\. Configuración de Python 3.11.

6\. Instalación de las dependencias necesarias para las pruebas E2E.

7\. Ejecución de las pruebas de extremo a extremo:



&#x20;  `pytest tests/test\_e2e.py -v`



8\. Consulta de los registros de Docker en caso de error.

9\. Apagado de los servicios al finalizar la ejecución.



\## 5. Integración de los servicios



Las pruebas de integración verifican la comunicación entre los siguientes componentes:



\*\*FastAPI → MLflow Model Serving → Modelo P2\*\*



Los servicios orquestados mediante Docker Compose son:



| Servicio | Puerto | Función |

|---|---:|---|

| MLflow | 5000 | Servidor de seguimiento de experimentos. |

| Model Serving | 5001 | Servicio de inferencia del modelo P2. |

| API | 8000 | Interfaz para recibir solicitudes de predicción. |



Se utilizan verificaciones de salud (\*healthchecks\*) y dependencias entre servicios para asegurar que los componentes estén disponibles antes de ejecutar las pruebas de integración.



\## 6. Correcciones de calidad de código



Durante la implementación se identificaron y corrigieron observaciones de Ruff, incluyendo:



\- Organización de bloques de importaciones.

\- Eliminación de importaciones no utilizadas.

\- Corrección de variables asignadas que no se utilizaban.

\- Ajustes de compatibilidad y estilo del código.

\- Normalización de criterios de revisión estática.



Posteriormente se ejecutaron nuevamente las pruebas para verificar que las correcciones no afectaran el funcionamiento del proyecto.



\## 7. Resultados obtenidos



La implementación se validó mediante GitHub Actions.



El pipeline completó satisfactoriamente los dos trabajos:



| Validación | Resultado |

|---|---|

| Revisión de calidad con Ruff | Aprobada |

| Pruebas automatizadas con Pytest | Aprobadas |

| Construcción de imágenes Docker | Aprobada |

| Inicio de servicios con Docker Compose | Aprobado |

| Pruebas de extremo a extremo | Aprobadas |

| Ejecución del pipeline completo | Exitosa |



La ejecución final observada en GitHub Actions tuvo una duración aproximada de 4 minutos y 8 segundos.



\## 8. Beneficios de la implementación



\- Automatización de las validaciones del proyecto.

\- Detección temprana de errores en los cambios realizados.

\- Estandarización de la revisión de calidad del código.

\- Verificación automática de las pruebas unitarias y de integración.

\- Validación de la construcción y comunicación entre los contenedores.

\- Mayor confiabilidad al integrar cambios al repositorio.



\## 9. Archivos relacionados



\- `.github/workflows/ci.yml`

\- `Containerfile`

\- `compose.yaml`

\- `requirements.txt`

\- `api/requirements.txt`

\- `models/p2/requirements.txt`

\- `tests/`



\## 10. Conclusión



Se implementó exitosamente un flujo de Integración Continua para el proyecto MiBici MLOps mediante GitHub Actions.



La solución automatiza la revisión de calidad del código, las pruebas automatizadas, la construcción de imágenes Docker y las pruebas de integración de extremo a extremo.



Con esta implementación se fortalece el proceso de desarrollo colaborativo y se establece una validación automática de los cambios realizados en el repositorio.



\*\*Nota de alcance:\*\* esta implementación cubre Integración Continua (CI), construcción de contenedores y pruebas de integración. No incluye todavía una etapa de despliegue automático a producción (CD).

