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
    RESULTADOS,
    exist_ok=True
)


# ---------------------------------------------------------
# COMPROBAR MODELO
# ---------------------------------------------------------

print("=" * 60)
print("CARGAR MODELO YOLOv8")
print("=" * 60)


if not os.path.exists(MODELO):

    print()
    print("No se encontró el modelo entrenado:")

    print(MODELO)

    print()
    print(
        "Debe ejecutar primero el script de entrenamiento."
    )

    raise SystemExit


# ---------------------------------------------------------
# CARGAR
# ---------------------------------------------------------

print()
print("Cargando modelo...")

modelo = YOLO(MODELO)


print()
print("Modelo cargado correctamente.")


# ---------------------------------------------------------
# INFORMACIÓN
# ---------------------------------------------------------

print()
print("Clases del modelo:")

try:

    print(modelo.names)

except Exception:

    print(
        "No fue posible mostrar las clases."
    )


# ---------------------------------------------------------
# REALIZAR UNA NUEVA PREDICCIÓN
# ---------------------------------------------------------

imagenes_disponibles = []

if os.path.exists(
    IMAGENES_PRUEBA
):

    for nombre in os.listdir(
        IMAGENES_PRUEBA
    ):

        if nombre.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            imagenes_disponibles.append(
                os.path.join(
                    IMAGENES_PRUEBA,
                    nombre
                )
            )


if len(imagenes_disponibles) > 0:

    print()
    print(
        "Realizando una nueva predicción..."
    )


    modelo.predict(

        source=imagenes_disponibles[0],

        imgsz=640,

        conf=0.25,

        save=True,

        project=RESULTADOS,

        name="modelo_cargado",

        exist_ok=True
    )


    print()
    print(
        "Predicción realizada correctamente."
    )

    print()
    print(
        "Resultado guardado en:"
    )

    print(
        os.path.join(
            RESULTADOS,
            "modelo_cargado"
        )
    )


else:

    print()
    print(
        "No hay imágenes de prueba."
    )

    print()
    print(
        "El modelo sí fue cargado correctamente."
    )


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print()
print("=" * 60)
print("PROCESO FINALIZADO")
print("=" * 60)