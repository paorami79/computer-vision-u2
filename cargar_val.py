from pathlib import Path
import urllib.request
import zipfile
import shutil

BASE_DIR = Path(__file__).resolve().parent

LABELS = BASE_DIR / "dataset" / "labels" / "val"
IMAGES = BASE_DIR / "dataset" / "images" / "val"
ZIP_FILE = BASE_DIR / "val2017.zip"

URL = "http://images.cocodataset.org/zips/val2017.zip"

IMAGES.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CARGANDO IMÁGENES DE VALIDACIÓN")
print("=" * 60)

labels = list(LABELS.glob("*.txt"))

print(f"Etiquetas VAL: {len(labels)}")

if not ZIP_FILE.exists():
    print("\nDescargando val2017.zip...")
    print("Es una descarga grande, espera a que termine.")
    urllib.request.urlretrieve(URL, ZIP_FILE)
    print("Descarga terminada.")
else:
    print("\nval2017.zip ya existe. Se utilizará el archivo existente.")

print("\nExtrayendo solamente las imágenes necesarias...")

nombres = {archivo.stem for archivo in labels}
copiadas = 0

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:

    archivos_zip = set(zip_ref.namelist())

    for nombre in nombres:

        archivo_zip = f"val2017/{nombre}.jpg"

        if archivo_zip in archivos_zip:

            destino = IMAGES / f"{nombre}.jpg"

            if not destino.exists():

                with zip_ref.open(archivo_zip) as origen:
                    with open(destino, "wb") as destino_file:
                        shutil.copyfileobj(origen, destino_file)

                copiadas += 1

print("\n" + "=" * 60)
print("RESULTADO")
print("=" * 60)

print(f"Imágenes nuevas copiadas: {copiadas}")
print(f"Imágenes VAL actuales: {len(list(IMAGES.glob('*.jpg')))}")
print(f"Carpeta: {IMAGES}")
