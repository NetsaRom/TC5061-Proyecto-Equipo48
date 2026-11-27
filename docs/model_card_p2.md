# Model Card — MiBici P2

## 1. Identificación del modelo

- Proyecto: MiBici MLOps
- Equipo: 48
- Modelo: mibici-equipo48-p2-riesgo
- Versión del artefacto desplegado: 1
- Tipo: Clasificación binaria
- Algoritmo documentado: Random Forest
- Framework: Scikit-learn
- Gestión del modelo: MLflow
- Despliegue: FastAPI y Docker Compose

## 2. Objetivo del modelo

Clasificar las estaciones del sistema MiBici
según su nivel de demanda, utilizando
características históricas de los viajes.

La clasificación constituye una aproximación
al riesgo operativo asociado con la demanda
de bicicletas.

El modelo no predice directamente la
disponibilidad futura de bicicletas.

## 3. Uso previsto

- Identificación de estaciones de alta demanda.
- Apoyo a decisiones operativas.
- Priorización de estaciones para su análisis.
- Apoyo a estrategias de redistribución.

El modelo debe utilizarse como herramienta
de apoyo y no como único mecanismo de decisión.

## 4. Datos

Fuente: registros históricos de viajes MiBici.

Periodos utilizados:

- Entrenamiento documentado: julio a septiembre.
- Referencia de monitoreo: julio a septiembre.
- Datos actuales de monitoreo: octubre.
- Registros de referencia: 470.
- Registros actuales: 470.

Los datos se agregan a nivel de estación.

## 5. Variables de entrada

| Variable | Descripción |
|---|---|
| promedio_duracion_salida | Duración promedio de los viajes de salida |
| pct_viajes_matutinos | Proporción de viajes realizados por la mañana |
| viajes_entrantes | Cantidad acumulada de viajes entrantes |

La API valida valores numéricos no negativos
y una proporción matutina entre 0 y 1.

## 6. Variable objetivo

Variable: alta_demanda.

- Clase 0: Demanda no alta.
- Clase 1: Alta demanda.

La etiqueta se construye utilizando un umbral
de demanda basado en la mediana de los viajes
de salida por estación.

## 7. Arquitectura del modelo

La documentación del modelo de clasificación
describe un Random Forest con:

- Número de árboles: 200.
- Balanceo de clases: balanced.
- Random state: 42.
- Proporción de prueba: 20%.
- División estratificada.

La correspondencia exacta de estos parámetros
con el artefacto actualmente desplegado
está pendiente de verificación.

## 8. Desempeño documentado

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9681 |
| F1-score | 0.9684 |
| ROC-AUC | 0.9968 |

Estos resultados proceden de la documentación
del modelo de clasificación.

No se ha confirmado su correspondencia exacta
con el Run ID del artefacto desplegado.

## 9. Evaluación A/B offline

Comparación documentada con datos de octubre:

| Métrica | Baseline | P2 Candidate 1 |
|---|---:|---:|
| Accuracy | 0.95957 | 0.96383 |
| Precision | 0.97705 | 0.97720 |
| Recall | 0.96129 | 0.96774 |
| F1-score | 0.96911 | 0.97245 |
| ROC-AUC | 0.98514 | 0.99297 |
| Inferencia por registro (ms) | 0.03837 | 0.03727 |

El candidato presenta mejoras en las métricas
evaluadas.

La comparación es offline y no constituye
un experimento A/B con tráfico productivo.

No se ha demostrado significancia estadística
ni se ha confirmado la promoción del candidato.

## 10. Despliegue

Arquitectura de inferencia:

Cliente -> FastAPI -> MLflow Model Serving
-> Modelo P2.

Servicios administrados mediante Docker Compose:

- MLflow Tracking.
- MLflow Model Serving.
- FastAPI.

Endpoints:

- GET /health
- POST /predict

El modelo desplegado se carga desde el
artefacto local models/p2.

## 11. Validación operativa

Se ejecutaron cinco pruebas E2E exitosas.

Las pruebas incluyen:

- Disponibilidad del servicio.
- Predicción válida.
- Validación de valores fuera de rango.
- Validación de campos faltantes.
- Validación de tipos de datos.

El pipeline de GitHub Actions ejecuta
validaciones de calidad, pruebas y
construcción de contenedores.

La integración continua fue verificada.
No se ha confirmado un despliegue continuo
hacia un ambiente productivo persistente.

## 12. Pruebas de carga

Herramienta: Locust.

Fecha de validación local: 26 de noviembre de 2026.

| Indicador | 20 usuarios | 50 usuarios |
|---|---:|---:|
| Duración configurada (s) | 60 | 60 |
| Solicitudes | 560 | 1393 |
| Errores | 0 | 0 |
| Throughput (RPS) | 9.62 | 23.28 |
| Latencia promedio (ms) | 38 | 32 |
| P95 (ms) | 65 | 70 |
| P99 (ms) | 130 | 100 |

Las pruebas demuestran estabilidad operativa
bajo las condiciones locales evaluadas.

No representan la capacidad máxima del sistema
ni garantizan el rendimiento en producción.

## 13. Monitoreo de Data Drift

Herramienta: Evidently.

Prueba estadística: Kolmogorov-Smirnov.

Nivel de significancia: 0.05.

Umbral de drift del dataset: 50%.

| Variable | Data Drift |
|---|---|
| promedio_duracion_salida | Detectado |
| pct_viajes_matutinos | No detectado |
| viajes_entrantes | Detectado |

Resultado global:

- Variables evaluadas: 3.
- Variables con drift: 2.
- Proporción de drift: 66.67%.
- Dataset Drift: Detectado.

La duración promedio disminuyó de 10.78 a 9.19,
equivalente a una reducción aproximada del 14.75%.

El p-value de pct_viajes_matutinos fue 0.147598.

Para viajes_entrantes, Evidently reportó un
p-value de 0 con la precisión mostrada.

El resultado de drift de promedio_duracion_salida
se infiere del resumen global del reporte.
Su p-value exacto queda pendiente de documentar.

La diferencia entre los periodos de referencia
(92 días) y actual (31 días) puede contribuir
al drift de variables de conteo acumulado.

El Data Drift no implica necesariamente
una degradación del desempeño predictivo.

## 14. FinOps

Se documentó una optimización de MLflow Tracking
mediante la reducción de cuatro workers a uno.

| Indicador | Original | Optimizado |
|---|---:|---:|
| Workers | 4 | 1 |
| Memoria (GiB) | 2.530 | 1.869 |

Reducción de memoria: 26.1%.

La validación local observó un consumo
de 1.904 GiB para MLflow Tracking.

No se observó una degradación relevante
del rendimiento de inferencia en las
pruebas documentadas.

No se ha demostrado un ahorro monetario real.

La capacidad de escritura concurrente
de MLflow Tracking requiere pruebas adicionales.

## 15. Versionado y linaje

Nombre del modelo:
mibici-equipo48-p2-riesgo

Versión declarada por el artefacto local: 1.

Run ID del artefacto desplegado:
0766e362faf644129dd54d9a2993380c

Model ID del artefacto desplegado:
m-c390896ac5bb4a9abfb743215b7c55e1

Existe evidencia histórica de registro
de una versión con alias champion:

Run ID:
36bf203639d6422789dac6f72e5f769b

Los identificadores de ambas evidencias
no coinciden.

Por lo tanto, queda pendiente reconciliar
el linaje del artefacto desplegado con
el registro original de MLflow.

La versión declarada por la API no constituye
por sí misma evidencia de promoción en Registry.

## 16. Riesgos y limitaciones

- La demanda histórica es una aproximación
  al riesgo operativo, no una medición directa
  de disponibilidad futura.
- Posibles efectos de estacionalidad.
- Diferencias en las ventanas temporales.
- Posible degradación por Data Drift.
- Posible fuga de información temporal si
  las variables no están disponibles antes
  del momento de predicción.
- Ausencia de validación productiva prolongada.
- Linaje del modelo pendiente de reconciliación.
- Pruebas de carga limitadas al entorno local.
- Evaluación A/B sin tráfico productivo.
- Impacto diferenciado por estación o zona
  geográfica pendiente de evaluación.

## 17. Estrategia de mantenimiento

Se recomienda:

1. Monitorear periódicamente las variables.
2. Comparar ventanas temporales equivalentes.
3. Evaluar calidad y disponibilidad de datos.
4. Medir desempeño con etiquetas recientes.
5. Investigar las causas del Data Drift.
6. Evaluar modelos candidatos.
7. Registrar versiones y métricas en MLflow.
8. Promover candidatos mediante criterios
   de aceptación previamente definidos.
9. Mantener mecanismos de rollback.

El reentrenamiento no debe activarse únicamente
por la detección de Data Drift.

## 18. Estado de validación

- Despliegue local: validado.
- Inferencia: validada.
- Pruebas E2E: validadas.
- Integración continua: validada.
- Pruebas de carga: validadas.
- Data Drift: evidencia verificada.
- FinOps: evidencia verificada.
- Evaluación A/B offline: documentada.
- Linaje completo: pendiente.
- Continuous Deployment: pendiente.

## 19. Referencias internas

- docs/modelo_clasificacion.md
- docs/fastapi-model-serving.md
- docs/p2-data-drift-monitoring.md
- docs/locust-monitoring-finops.md
- reports/p2_evidently_drift_report.html
- reports/evidencias/resultados_ab_p2_octubre_c1.csv
- models/p2/MLmodel
- models/p2/registered_model_meta
- tests/test_e2e.py
- tests/locustfile.py
- .github/workflows/ci.yml

## 20. Responsabilidad

Equipo 48 — Proyecto Integrador MLOps.

Fecha de elaboración: 26 de noviembre de 2026.

Estado: documentación técnica para revisión
y entrega académica.

