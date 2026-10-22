"""Punto de entrada para preparar los datos del proyecto MiBici."""

import subprocess
import sys


def ingest() -> None:
    """Ejecuta la ingesta y genera las características del proyecto."""

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mibici_mlops.dataset",
            "procesar-periodos",
        ],
        check=True,
    )

    subprocess.run(
        [sys.executable, "-m", "mibici_mlops.features"],
        check=True,
    )


if __name__ == "__main__":
    ingest()
