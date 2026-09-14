import os
from ultralytics import YOLO


# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODELO = os.path.join(
    BASE_DIR,
    "modelos",
    "yolov8_coco",
    "weights",
    "best.pt"
)


IMAGENES_PRUEBA = os.path.join(
    BASE_DIR,
    "imagenes_prueba"
)


RESULTADOS = os.path.join(
    BASE_DIR,
    "resultados"
)


os.makedirs(
    IMAGENES_PRUEBA,
    exist_ok=True
)

os.makedirs(
    RESULTADOS,
    exist_ok=True
)


# ---------------------------------------------------------
# VERIFICAR MODELO
# ---------------------------------------------------------

print("=" * 60)
print("INFERENCIA YOLOv8")
print("=" * 60)


if not os.path.exists(MODELO):

    print()
    print("ERROR: no existe el modelo.")

    print(MODELO)

    print()
    print(
        "Ejecute primero el entrenamiento."
    )

    raise SystemExit


# ---------------------------------------------------------
# CARGAR MODELO
# ---------------------------------------------------------

print()
print("Cargando modelo...")

modelo = YOLO(MODELO)


# ---------------------------------------------------------
# BUSCAR IMÁGENES
# ---------------------------------------------------------

imagenes = []

for nombre in os.listdir(
    IMAGENES_PRUEBA
):

    if nombre.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        imagenes.append(
            os.path.join(
                IMAGENES_PRUEBA,
                nombre
            )
        )


if len(imagenes) == 0:

    print()
    print(
        "No se encontraron imágenes."
    )

    print()
    print(
        "Coloque imágenes JPG o PNG en:"
    )

    print(IMAGENES_PRUEBA)

    raise SystemExit


# ---------------------------------------------------------
# PREDICCIÓN
# ---------------------------------------------------------

print()
print(
    f"Imágenes encontradas: {len(imagenes)}"
)

print()
print("Ejecutando detección...")


modelo.predict(

    source=IMAGENES_PRUEBA,

    imgsz=640,

    conf=0.25,

    save=True,

    project=RESULTADOS,

    name="predicciones",

    exist_ok=True
)


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print()
print("=" * 60)
print("INFERENCIA FINALIZADA")
print("=" * 60)

print()
print(
    "Las imágenes con detecciones se encuentran en:"
)

print(
    os.path.join(
        RESULTADOS,
        "predicciones"
    )
)