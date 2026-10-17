import hashlib

from mibici_mlops.dataset import calcular_sha256


def test_calcular_sha256(tmp_path):
    """Verifica que el hash SHA-256 se calcula correctamente."""

    # Crear un archivo temporal
    archivo = tmp_path / "prueba.txt"
    archivo.write_text("MiBici Equipo 48", encoding="utf-8")

    # Calcular el hash esperado
    esperado = hashlib.sha256(
        b"MiBici Equipo 48"
    ).hexdigest()

    # Calcular el hash con nuestra función
    resultado = calcular_sha256(archivo)

    # Comprobar que coinciden
    assert resultado == esperado
    assert len(resultado) == 64
