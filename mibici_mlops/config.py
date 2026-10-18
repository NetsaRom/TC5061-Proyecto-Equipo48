from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# ============================================================
# Configuración del proyecto MiBici - Equipo 48
# ============================================================

TEAM_ID = 48
RANDOM_SEED = 42

FECHA_INICIO = "2026-07"
FECHA_FIN = "2026-09"

# Detectar y descargar los catálogos disponibles
DESCARGAR_CATALOGOS = True


def generar_periodos(inicio: str, fin: str) -> list[tuple[int, int]]:
    """Genera los meses entre dos fechas YYYY-MM."""
    from datetime import date

    anio_inicio, mes_inicio = map(int, inicio.split("-"))
    anio_fin, mes_fin = map(int, fin.split("-"))

    actual = date(anio_inicio, mes_inicio, 1)
    limite = date(anio_fin, mes_fin, 1)

    if actual > limite:
        raise ValueError("La fecha inicial no puede superar la final.")

    periodos = []

    while actual <= limite:
        periodos.append((actual.year, actual.month))

        if actual.month == 12:
            actual = date(actual.year + 1, 1, 1)
        else:
            actual = date(actual.year, actual.month + 1, 1)

    return periodos


PERIODOS_DATOS = generar_periodos(FECHA_INICIO, FECHA_FIN)