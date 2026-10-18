
from pathlib import Path

import polars as pl
import typer
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from loguru import logger

from mibici_mlops.config import (PROCESSED_DATA_DIR, DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR, PERIODOS_DATOS, DESCARGAR_CATALOGOS, FECHA_INICIO, FECHA_FIN, generar_periodos, PROJ_ROOT)

import hashlib
import json
from datetime import datetime, timezone

app = typer.Typer()

def calcular_sha256(ruta_archivo: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""

    sha256 = hashlib.sha256()

    with open(ruta_archivo, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha256.update(bloque)

    return sha256.hexdigest()

def generar_manifiesto(rutas_archivos: list[Path], ruta_salida: Path) -> None:
    """Genera un manifiesto JSON con metadatos y hashes SHA-256."""

    archivos = []

    for ruta in rutas_archivos:
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el archivo: {ruta}")

        try:
            ruta_registro = ruta.resolve().relative_to(PROJ_ROOT.resolve()).as_posix()
        except ValueError:
            ruta_registro = ruta.as_posix()
        registros = None

        if ruta.suffix.lower() == ".parquet":
            registros = pl.scan_parquet(ruta).select(pl.len()).collect().item()

        archivos.append({
            "nombre": ruta.name,
            "ruta": ruta_registro,
            "tamano_bytes": ruta.stat().st_size,
            "registros": registros,
            "sha256": calcular_sha256(ruta),
        })

    manifiesto = {
        "equipo": 48,
        "version_manifiesto": "1.0",
        "fecha_generacion_utc": datetime.now(timezone.utc).isoformat(),
        "archivos": archivos,
    }

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        json.dump(manifiesto, archivo, indent=4, ensure_ascii=False)

def leer_viajes(ruta: Path) -> pl.DataFrame:
    """Lee el archivo CSV original de MiBici."""

    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    return pl.read_csv(
        ruta,
        null_values=["NULL", "NA", ""],
        try_parse_dates=True,
        infer_schema_length=10000,
    )

def leer_estaciones(ruta: Path) -> pl.DataFrame:
    """Lee y valida el catálogo oficial de estaciones MiBici."""

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe el catálogo de estaciones: {ruta}"
        )

    df = pl.read_csv(ruta)

    columnas = [
        "id",
        "name",
        "latitude",
        "longitude",
        "dpcapacity",
    ]

    faltantes = set(columnas) - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas en el catálogo: {sorted(faltantes)}"
        )

    if df["id"].is_duplicated().any():
        raise ValueError("El catálogo contiene IDs duplicados")

    if df.select(pl.all().is_null().any()).row(0).count(True) > 0:
        raise ValueError("El catálogo contiene valores nulos")

    return df.select(columnas)

def obtener_enlaces_csv() -> dict[str, str]:
    """Obtiene los enlaces CSV oficiales de MiBici usando curl."""

    import subprocess

    url_base = "https://mibici.net"
    url_pagina = f"{url_base}/es/datos-abiertos/"

    resultado = subprocess.run(
        [
            "curl",
            "-fLsS",
            "--max-time", "60",
            url_pagina,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    soup = BeautifulSoup(resultado.stdout, "html.parser")

    enlaces = {}

    for elemento in soup.find_all("a", href=True):
        href = elemento["href"]
        url_completa = urljoin(url_pagina, href)

        if url_completa.lower().split("?")[0].endswith(".csv"):
            nombre_archivo = url_completa.split("/")[-1].split("?")[0]
            enlaces[nombre_archivo] = url_completa

    return enlaces

def descargar_archivo(
    url: str,
    ruta_destino: Path,
) -> Path:
    """Descarga un archivo CSV si todavía no existe."""

    import subprocess

    ruta_destino.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if ruta_destino.is_file() and ruta_destino.stat().st_size > 0:
        logger.info(f"Archivo existente: {ruta_destino.name}")
        return ruta_destino

    logger.info(f"Descargando: {ruta_destino.name}")

    ruta_temporal = ruta_destino.with_suffix(
        ruta_destino.suffix + ".part"
    )

    try:
        subprocess.run(
            [
                "curl",
                "-fLsS",
                "--retry", "3",
                "--connect-timeout", "30",
                "--max-time", "300",
                url,
                "-o", str(ruta_temporal),
            ],
            check=True,
        )

        if ruta_temporal.stat().st_size == 0:
            raise ValueError("El archivo descargado está vacío")

        ruta_temporal.replace(ruta_destino)

    finally:
        ruta_temporal.unlink(missing_ok=True)

    logger.success(f"Descarga completada: {ruta_destino.name}")

    return ruta_destino

def procesar_mes(anio: int, mes: int) -> None:
    """Descarga y procesa los viajes y el catálogo de un mes."""

    periodo = f"{anio}_{mes:02d}"

    nombre_viajes = f"datos_abiertos_{periodo}.csv"

    enlaces = obtener_enlaces_csv()

    if nombre_viajes not in enlaces:
        raise FileNotFoundError(
            f"No se encontró {nombre_viajes} en la página de MiBici"
        )

    ruta_viajes = descargar_archivo(
        enlaces[nombre_viajes],
        RAW_DATA_DIR / nombre_viajes,
    )

    viajes = limpiar_viajes(leer_viajes(ruta_viajes))

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    viajes.write_parquet(
        PROCESSED_DATA_DIR / f"viajes_{periodo}.parquet"
    )

    
    logger.success(
        f"Periodo {periodo}: "
        f"{viajes.height:,} viajes procesados"
    )

@app.command("procesar-periodos")
def procesar_periodos(
    inicio: str = typer.Option(
        FECHA_INICIO,
        "--inicio",
        help="Mes inicial en formato YYYY-MM",
    ),
    fin: str = typer.Option(
        FECHA_FIN,
        "--fin",
        help="Mes final en formato YYYY-MM",
    ),
) -> None:
    """Procesa los meses solicitados y los catálogos disponibles."""

    periodos = generar_periodos(inicio, fin)

    enlaces = obtener_enlaces_csv()

    logger.info(
        f"Iniciando procesamiento de {len(periodos)} meses "
        f"({inicio} a {fin})"
    )

    for anio, mes in periodos:
        procesar_mes(anio, mes)

    # Detectar y procesar los catálogos disponibles
    if DESCARGAR_CATALOGOS:
        catalogos = {
            nombre: url
            for nombre, url in enlaces.items()
            if nombre.lower().startswith("nomenclatura")
            and nombre.lower().endswith(".csv")
        }

        for nombre, url in sorted(catalogos.items()):
            ruta = descargar_archivo(
                url,
                EXTERNAL_DATA_DIR / nombre,
            )

            estaciones = leer_estaciones(ruta)

            nombre_parquet = Path(nombre).stem + ".parquet"

            PROCESSED_DATA_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            estaciones.write_parquet(
                PROCESSED_DATA_DIR / nombre_parquet
            )

            logger.success(
                f"Catálogo procesado: {nombre} "
                f"({estaciones.height:,} estaciones)"
            )

        # Reunir los CSV y Parquet de los periodos procesados
    archivos_manifiesto = []

    for anio, mes in periodos:
        periodo = f"{anio}_{mes:02d}"

        archivos_manifiesto.extend([
            RAW_DATA_DIR / f"datos_abiertos_{periodo}.csv",
            PROCESSED_DATA_DIR / f"viajes_{periodo}.parquet",
        ])

    # Incluir los catálogos procesados durante esta ejecución
    if DESCARGAR_CATALOGOS:
        for nombre in sorted(catalogos):
            archivos_manifiesto.extend([
                EXTERNAL_DATA_DIR / nombre,
                PROCESSED_DATA_DIR / f"{Path(nombre).stem}.parquet",
            ])

    # Generar manifiesto con los archivos existentes
    generar_manifiesto(
        rutas_archivos=archivos_manifiesto,
        ruta_salida=DATA_DIR / "manifest.json",
    )

    logger.success("Manifiesto SHA-256 generado correctamente")

    logger.success("Procesamiento de periodos finalizado")

def limpiar_viajes(df: pl.DataFrame) -> pl.DataFrame:
    """Elimina duplicados exactos y prepara los viajes."""

    columnas = [
        "Viaje_Id",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]

    faltantes = set(columnas) - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(faltantes)}"
        )

    # Eliminar registros completamente duplicados
    df_limpio = df.unique(maintain_order=True)

    # Conservar las columnas operativas
    df_limpio = df_limpio.select(columnas)

    # Calcular duración en minutos
    df_limpio = df_limpio.with_columns(
        (
            (
                pl.col("Fin_del_viaje")
                - pl.col("Inicio_del_viaje")
            ).dt.total_seconds()
            / 60
        ).alias("duracion_min")
    )

    # Marcar duraciones anómalas sin eliminarlas
    df_limpio = df_limpio.with_columns(
        (
            (pl.col("duracion_min") <= 0)
            | (pl.col("duracion_min") > 1440)
        ).alias("duracion_atipica")
    )

    return df_limpio



@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "datos_abiertos_2026_09.csv",
    output_path: Path = PROCESSED_DATA_DIR / "viajes_2026_09.parquet",
):
    """Procesa viajes MiBici y genera un archivo Parquet."""

    logger.info(f"Leyendo archivo: {input_path}")

    df = leer_viajes(input_path)
    logger.info(f"Registros originales: {df.height:,}")

    df_limpio = limpiar_viajes(df)
    logger.info(f"Registros procesados: {df_limpio.height:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_limpio.write_parquet(output_path)

    logger.success(f"Archivo generado: {output_path}")


if __name__ == "__main__":
    app()
