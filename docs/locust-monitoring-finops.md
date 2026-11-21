# Pruebas de carga, monitoreo y optimización FinOps — MiBici P2

## 1. Objetivo

Evaluar el rendimiento de la API REST del modelo P2 de MiBici mediante pruebas de carga con Locust, monitorear el consumo de recursos de los microservicios desplegados con Docker Compose e identificar oportunidades de optimización de infraestructura bajo principios FinOps.

Los objetivos específicos son:

- Validar el comportamiento de la API ante diferentes niveles de concurrencia.
- Medir latencia, throughput y tasa de errores.
- Analizar el consumo de CPU y memoria de los microservicios.
- Identificar posibles cuellos de botella.
- Implementar y evaluar una optimización de recursos.
- Documentar resultados reproducibles y limitaciones del experimento.

## 2. Arquitectura evaluada

La solución está compuesta por tres servicios orquestados mediante Docker Compose.

| Microservicio | Función | Puerto interno |
|---|---|---|
| FastAPI | Exponer el endpoint `/predict` | 8000 |
| MLflow Model Serving | Ejecutar inferencias del modelo P2 | 5001 |
| MLflow Tracking | Administrar experimentos, métricas y artefactos | 5000 |

El flujo de inferencia es:

```text
Cliente / Locust
       |
       v
FastAPI (/predict)
       |
       v
MLflow Model Serving
       |
       v
Modelo P2
       |
       v
Respuesta JSON
```

MLflow Tracking forma parte de la infraestructura, pero no participa directamente en cada solicitud de inferencia evaluada.

### 2.1. Inicio de los servicios

Desde la raíz del proyecto:

```bash
docker compose up -d --wait
```

En macOS, si el puerto 5000 está ocupado:

```bash
MLFLOW_PORT=5050 docker compose up -d --wait
```

En este caso, los servicios se encuentran disponibles en:

- API REST: `http://localhost:8000`
- MLflow Tracking: `http://localhost:5050` cuando se utiliza la configuración alternativa.
- Model Serving: puerto 5001 dentro de la red Docker.

Para verificar los servicios:

```bash
docker compose ps
```

Los tres servicios deben encontrarse en estado saludable.

## 3. Pruebas de carga con Locust

### 3.1. Herramienta utilizada

Se utilizó Locust 2.47.0 para simular usuarios concurrentes realizando solicitudes HTTP POST al endpoint `/predict`.

La implementación se encuentra en:

`tests/locustfile.py`

El usuario simulado espera entre 1 y 3 segundos entre solicitudes.

Cada petición envía un conjunto de características al modelo P2 y verifica:

1. Que el servidor responda con HTTP 200.
2. Que la respuesta sea un JSON válido.
3. Que el campo `prediccion` contenga una clase válida: 0 o 1.

### 3.2. Instalación de Locust

Crear un entorno virtual:

```bash
python -m venv .venv-locust
```

Activarlo en macOS o Linux:

```bash
source .venv-locust/bin/activate
```

Instalar la versión utilizada:

```bash
python -m pip install locust==2.47.0
```

Verificar que Locust reconozca el usuario de prueba:

```bash
locust -f tests/locustfile.py --list
```

### 3.3. Escenarios ejecutados

Se realizaron cinco pruebas:

| Escenario | Usuarios concurrentes | Duración | Tasa de incorporación |
|---|---:|---|---:|
| Piloto | 5 | 1 minuto | 1 usuario/s |
| Carga intermedia | 20 | 1 minuto | 2 usuarios/s |
| Carga alta | 50 | 1 minuto | 5 usuarios/s |
| Monitoreo base | 50 | 2 minutos | 5 usuarios/s |
| Configuración optimizada | 50 | 2 minutos | 5 usuarios/s |

Ejemplo para reproducir la prueba de monitoreo:

```bash
locust -f tests/locustfile.py \
  --headless \
  --host http://localhost:8000 \
  --users 50 \
  --spawn-rate 5 \
  --run-time 2m \
  --csv reports/locust_50_monitoreo
```

Para evitar sobrescribir evidencias existentes, puede utilizarse otro prefijo CSV al repetir el experimento.

### 3.4. Resultados obtenidos

| Escenario | Solicitudes | Errores | RPS | Latencia promedio | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| 5 usuarios, 1 minuto | 147 | 0 | 2.46 | 41 ms | 55 ms | 74 ms |
| 20 usuarios, 1 minuto | 538 | 0 | 9.01 | 38 ms | 62 ms | 77 ms |
| 50 usuarios, 1 minuto | 1,377 | 0 | 23.03 | 34 ms | 74 ms | 110 ms |
| 50 usuarios, 2 minutos | 2,836 | 0 | 23.76 | 32.68 ms | 67 ms | 110 ms |
| 50 usuarios, 2 minutos, configuración optimizada | 2,862 | 0 | 23.89 | ~33 ms | 65 ms | 110 ms |

**Interpretación:**

- No se registraron solicitudes fallidas en los escenarios evaluados.
- El throughput aumentó conforme se incrementó la concurrencia.
- En las pruebas de 50 usuarios y 2 minutos se alcanzaron aproximadamente 23.8 solicitudes por segundo.
- La latencia P95 se mantuvo por debajo de 75 ms.
- La configuración optimizada no mostró una degradación apreciable del rendimiento de inferencia en este experimento.

Los resultados corresponden a un entorno local y no garantizan el mismo comportamiento en producción.

## 4. Monitoreo de infraestructura

### 4.1. Herramienta utilizada

Se utilizó Docker Stats para observar el consumo de CPU y memoria de los contenedores.

Comando para obtener una medición:

```bash
docker stats --no-stream
```

También se utilizó `docker stats` de manera continua durante las pruebas.

### 4.2. Consumo antes de las pruebas de carga

| Microservicio | CPU | Memoria |
|---|---:|---:|
| MLflow Tracking | 2.37% | 2.533 GiB |
| FastAPI | 0.19% | 87.18 MiB |
| MLflow Model Serving | 0.20% | 439.5 MiB |

### 4.3. Consumo observado durante la prueba de 50 usuarios

| Microservicio | CPU | Memoria |
|---|---:|---:|
| MLflow Tracking | 0.79% | 2.533 GiB |
| FastAPI | 3.96% | 87.79 MiB |
| MLflow Model Serving | 43.47% | 440.4 MiB |

### 4.4. Interpretación del monitoreo

MLflow Model Serving presentó el mayor consumo de CPU durante las inferencias, lo cual es consistente con su responsabilidad de ejecutar el modelo.

FastAPI incrementó moderadamente su utilización de CPU al procesar las solicitudes HTTP.

MLflow Tracking presentó el mayor consumo de memoria, incluso cuando no recibía directamente las solicitudes de predicción.

Es importante considerar que los porcentajes de CPU corresponden a mediciones instantáneas, no a promedios de toda la prueba.

Estas observaciones motivaron un experimento para reducir el consumo de memoria del servidor de tracking.

## 5. Experimento de optimización FinOps

### 5.1. Problema identificado

Durante el monitoreo se observó que MLflow Tracking consumía aproximadamente 2.53 GiB de memoria.

La configuración original utilizaba cuatro workers HTTP de Uvicorn, valor predeterminado del servidor MLflow en el entorno evaluado.

Para una ejecución local de desarrollo, se planteó evaluar si un solo worker permitía reducir el consumo de recursos sin afectar el funcionamiento observado de la solución.

### 5.2. Hipótesis

Reducir de cuatro a un worker HTTP de MLflow Tracking disminuirá el consumo de memoria del contenedor, manteniendo operativos los microservicios y sin producir una degradación apreciable del endpoint de inferencia bajo la carga evaluada.

### 5.3. Cambio implementado

Se modificó exclusivamente el comando del servicio `mlflow` en `compose.yaml`, agregando:

```yaml
- --workers
- "1"
```

La configuración fue validada mediante:

```bash
MLFLOW_PORT=5050 docker compose config --quiet
```

Posteriormente, se recreó únicamente el servicio MLflow:

```bash
MLFLOW_PORT=5050 docker compose up -d --no-deps --force-recreate mlflow
```

Se verificó que los tres servicios continuaran saludables.

### 5.4. Resultados de memoria

| Configuración | Memoria de MLflow |
|---|---:|
| Configuración original, 4 workers | 2.530 GiB |
| Configuración optimizada, 1 worker, antes de carga | 1.868 GiB |
| Configuración optimizada, 1 worker, después de carga | 1.869 GiB |

La reducción observada fue:

**Memoria liberada = 2.530 - 1.869 = 0.661 GiB**

**Reducción porcentual = (0.661 / 2.530) × 100 = 26.1%**

Esto representa aproximadamente 677 MiB menos de memoria utilizada por el contenedor MLflow Tracking.

### 5.5. Validación de rendimiento posterior

Después de implementar la optimización, se repitió una prueba de 50 usuarios durante 2 minutos.

| Indicador | Antes | Después |
|---|---:|---:|
| Solicitudes | 2,836 | 2,862 |
| Errores | 0 | 0 |
| Throughput | 23.76 RPS | 23.89 RPS |
| Latencia promedio | 32.68 ms | ~33 ms |
| P95 | 67 ms | 65 ms |
| P99 | 110 ms | 110 ms |

La diferencia de throughput fue pequeña y no permite afirmar una mejora de rendimiento estadísticamente significativa.

Sin embargo, no se observó una degradación relevante del endpoint `/predict` durante la prueba posterior.

**Limitación importante:** estas pruebas validan el rendimiento de inferencia, no la capacidad del servidor MLflow Tracking para recibir registros concurrentes de experimentos. Se requieren pruebas específicas de tracking antes de adoptar la configuración en producción.

## 6. Análisis FinOps

### 6.1. Interpretación de la optimización

La reducción de memoria constituye una mejora en la eficiencia de recursos.

No obstante, liberar memoria en Docker no genera automáticamente un ahorro monetario, especialmente cuando los contenedores continúan ejecutándose en la misma computadora o instancia.

Para materializar un ahorro económico, sería necesario demostrar que el menor consumo permite utilizar infraestructura de menor costo, consolidar servicios o evitar ampliaciones de capacidad.

### 6.2. Modelo para estimar costos

Una estimación mensual de infraestructura puede expresarse como:

```text
Costo mensual =
  Costo de cómputo
+ Costo de almacenamiento
+ Costo de transferencia de datos
+ Costo de monitoreo
+ Otros servicios asociados
```

El costo unitario de inferencia puede calcularse mediante:

```text
Costo por predicción =
Costo mensual atribuible al servicio de inferencia
/
Número de predicciones exitosas del mes
```

Para calcular un costo monetario real deben definirse:

- Proveedor de nube y región.
- Tipo y tamaño de instancia.
- Precio por hora.
- Horas de operación mensuales.
- Número mensual de predicciones.
- Costos de almacenamiento y transferencia.
- Recursos reservados y utilizados.

### 6.3. Recomendación FinOps

Mantener la configuración de un worker como alternativa de desarrollo local mientras se realizan pruebas adicionales de MLflow Tracking.

Antes de utilizarla en producción, comparar el rendimiento de ambas configuraciones con cargas representativas y evaluar el costo total de infraestructura.

## 7. Limitaciones del experimento

Las pruebas realizadas tienen las siguientes limitaciones:

1. Se ejecutaron en una computadora local, no en infraestructura productiva.
2. No se realizaron múltiples repeticiones de cada escenario para calcular intervalos de confianza.
3. El monitoreo de CPU se basó en observaciones puntuales.
4. No se evaluó la carga concurrente de escritura en MLflow Tracking.
5. No se establecieron límites explícitos de CPU o memoria para los contenedores