import os
import json
import shutil


# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

ANNOTATIONS_DIR = os.path.join(
    DATASET_DIR,
    "annotations"
)

LABELS_TRAIN = os.path.join(
    DATASET_DIR,
    "labels",
    "train"
)

LABELS_VAL = os.path.join(
    DATASET_DIR,
    "labels",
    "val"
)

IMAGES_TRAIN = os.path.join(
    DATASET_DIR,
    "images",
    "train"
)

IMAGES_VAL = os.path.join(
    DATASET_DIR,
    "images",
    "val"
)


os.makedirs(LABELS_TRAIN, exist_ok=True)

os.makedirs(LABELS_VAL, exist_ok=True)

os.makedirs(IMAGES_TRAIN, exist_ok=True)

os.makedirs(IMAGES_VAL, exist_ok=True)


# ---------------------------------------------------------
# CATEGORÍAS
# ---------------------------------------------------------

CATEGORIAS = {

    1: 0,       # person
    3: 1,       # car
    18: 2       # dog
}


NOMBRES = [
    "person",
    "car",
    "dog"
]


# ---------------------------------------------------------
# CONVERSIÓN DE UNA ANOTACIÓN
# ---------------------------------------------------------

def convertir_caja(bbox, ancho, alto):

    x, y, w, h = bbox

    x_centro = x + (w / 2)

    y_centro = y + (h / 2)

    x_centro = x_centro / ancho

    y_centro = y_centro / alto

    w = w / ancho

    h = h / alto

    return (
        x_centro,
        y_centro,
        w,
        h
    )


# ---------------------------------------------------------
# PROCESAMIENTO DEL JSON
# ---------------------------------------------------------

def procesar_json(archivo_json, carpeta_labels):

    print()
    print("Procesando:")
    print(archivo_json)

    with open(
        archivo_json,
        "r",
        encoding="utf-8"
    ) as archivo:

        datos = json.load(archivo)


    imagenes = {}

    for imagen in datos["images"]:

        imagenes[imagen["id"]] = imagen


    etiquetas = {}


    for anotacion in datos["annotations"]:

        categoria = anotacion["category_id"]

        if categoria not in CATEGORIAS:

            continue


        imagen_id = anotacion["image_id"]

        imagen = imagenes[imagen_id]

        ancho = imagen["width"]

        alto = imagen["height"]

        bbox = anotacion["bbox"]


        valores = convertir_caja(
            bbox,
            ancho,
            alto
        )


        clase = CATEGORIAS[categoria]


        linea = (
            f"{clase} "
            f"{valores[0]:.6f} "
            f"{valores[1]:.6f} "
            f"{valores[2]:.6f} "
            f"{valores[3]:.6f}"
        )


        if imagen_id not in etiquetas:

            etiquetas[imagen_id] = []


        etiquetas[imagen_id].append(linea)


    contador = 0


    for imagen_id, lineas in etiquetas.items():

        imagen = imagenes[imagen_id]

        nombre = os.path.splitext(
            imagen["file_name"]
        )[0]


        archivo_txt = os.path.join(
            carpeta_labels,
            nombre + ".txt"
        )


        with open(
            archivo_txt,
            "w",
            encoding="utf-8"
        ) as archivo:

            for linea in lineas:

                archivo.write(
                    linea + "\n"
                )


        contador += 1


    print(
        f"Archivos YOLO generados: {contador}"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

print("=" * 60)
print("CONVERSIÓN COCO → YOLO")
print("=" * 60)


train_json = os.path.join(
    ANNOTATIONS_DIR,
    "instances_train2017.json"
)

val_json = os.path.join(
    ANNOTATIONS_DIR,
    "instances_val2017.json"
)


if os.path.exists(train_json):

    procesar_json(
        train_json,
        LABELS_TRAIN
    )

else:

    print()
    print("No se encontró:")
    print(train_json)


if os.path.exists(val_json):

    procesar_json(
        val_json,
        LABELS_VAL
    )

else:

    print()
    print("No se encontró:")
    print(val_json)


print()
print("=" * 60)
print("CONVERSIÓN FINALIZADA")
print("=" * 60)

print()
print("Clases utilizadas:")

for indice, nombre in enumerate(NOMBRES):

    print(
        f"{indice} = {nombre}"
    )

