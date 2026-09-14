import os
import zipfile
import urllib.request
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

LABELS_VAL = BASE_DIR / "dataset" / "labels" / "val"
IMAGES_VAL = BASE_DIR / "dataset" / "images" / "val"

ZIP_VAL = BASE_DIR / "val2017.zip"

URL_VAL = "http://images.cocodataset.org/zips/val2017.zip"

# Crear carpeta destino
IMAGES_VAL.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DESCARGA DE IMÁGENES VAL - COCO 2017")
print("=" * 60)

# ============================================================
# 1. OBTENER NOMBRES DE LAS IMÁGENES NECESARIAS
# ============================================================

labels = list(LABELS_VAL.glob("*.txt"))

print(f"\nEtiquetas VAL encontradas: {len(labels)}")

if len(labels) == 0:
    print("ERROR: No existen etiquetas en:")
    print(LABELS_VAL)
    exit()

# ============================================================
# 2. DESCARGAR ZIP DE VAL SI NO EXISTE
# ============================================================

if not ZIP_VAL.exists():

    print("\nDescargando val2017.zip...")
    print("Esta descarga puede tardar.")

    urllib.request.urlretrieve(
        URL_VAL,
        ZIP_VAL
    )

    print("\nDescarga terminada.")

else:

    print("\nval2017.zip ya existe.")
    print("No se volverá a descargar.")

# ============================================================
# 3. EXTRAER SOLAMENTE LAS IMÁGENES NECESARIAS
# ============================================================

print("\nBuscando imágenes correspondientes a las etiquetas...")

contador = 0

with zipfile.ZipFile(ZIP_VAL, "r") as zip_ref:

    archivos = zip_ref.namelist()

    for label in labels:

        nombre = label.stem + ".jpg"

        ruta_zip = "val2017/" + nombre

        if ruta_zip in archivos:

            destino = IMAGES_VAL / nombre

            if not destino.exists():

                with zip_ref.open(ruta_zip) as origen:
                    with open(destino, "wb") as salida:
                        salida.write(origen.read())

                contador += 1

print("\n" + "=" * 60)
print("PROCESO TERMINADO")
print("=" * 60)

print(f"Imágenes copiadas: {contador}")
print(f"Carpeta VAL:")
print(IMAGES_VAL)

# ============================================================
# 4. VERIFICACIÓN
# ============================================================

imagenes = list(IMAGES_VAL.glob("*.jpg"))

print(f"\nTotal imágenes VAL actualmente: {len(imagenes)}")

if len(imagenes) == 0:
    print("\nERROR: No se encontraron imágenes VAL.")

else:
    print("\nOK: Ya existen imágenes en VAL.")