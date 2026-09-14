import os
import cv2

import albumentations as A


# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


INPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "images",
    "train"
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "imagenes_procesadas"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# TRANSFORMACIONES
# ---------------------------------------------------------

transformacion = A.Compose(
    [
        A.Resize(
            height=640,
            width=640
        ),

        A.HorizontalFlip(
            p=0.5
        ),

        A.RandomBrightnessContrast(
            p=0.3
        )
    ]
)


# ---------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------

print("=" * 60)
print("AUMENTO DE DATOS CON ALBUMENTATIONS")
print("=" * 60)


if not os.path.exists(INPUT_DIR):

    print()
    print("No existe la carpeta:")
    print(INPUT_DIR)

    print()
    print(
        "Primero debe preparar las imágenes del dataset."
    )

    raise SystemExit


archivos = os.listdir(INPUT_DIR)

contador = 0


for nombre in archivos:

    if not nombre.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        continue


    ruta = os.path.join(
        INPUT_DIR,
        nombre
    )


    imagen = cv2.imread(ruta)


    if imagen is None:

        print(
            f"No se pudo leer: {nombre}"
        )

        continue


    resultado = transformacion(
        image=imagen
    )


    imagen_transformada = resultado[
        "image"
    ]


    salida = os.path.join(
        OUTPUT_DIR,
        "aug_" + nombre
    )


    cv2.imwrite(
        salida,
        imagen_transformada
    )


    contador += 1


print()
print("=" * 60)

print(
    f"Imágenes procesadas: {contador}"
)

print("=" * 60)

print()
print(
    "Las imágenes transformadas se encuentran en:"
)

print(OUTPUT_DIR)