from locust import HttpUser, between, task


class MiBiciUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predecir_demanda(self):
        payload = {
            "promedio_duracion_salida": 9.2547,
            "pct_viajes_matutinos": 0.2379,
            "viajes_entrantes": 1452,
        }

        with self.client.post(
            "/predict",
            json=payload,
            name="/predict",
            timeout=15,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                return

            try:
                resultado = response.json()
            except ValueError:
                response.failure("La respuesta no es JSON válido")
                return

            if resultado.get("prediccion") not in (0, 1):
                response.failure("La predicción no es válida")
