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


DATA_YAML = os.path.join(
    BASE_DIR,
    "data.yaml"
)


# ---------------------------------------------------------
# VERIFICAR MODELO
# ---------------------------------------------------------

print("=" * 60)
print("EVALUACIÓN DEL MODELO YOLOv8")
print("=" * 60)


if not os.path.exists(MODELO):

    print()
    print("ERROR:")
    print("No se encontró el modelo:")
    print(MODELO)

    print()
    print(
        "Debe ejecutar primero 05_entrenamiento.py"
    )

    raise SystemExit


# ---------------------------------------------------------
# CARGAR MODELO
# ---------------------------------------------------------

print()
print("Cargando modelo...")

modelo = YOLO(MODELO)


# ---------------------------------------------------------
# VALIDACIÓN
# ---------------------------------------------------------

print()
print("Ejecutando validación...")

metricas = modelo.val(

    data=DATA_YAML,

    imgsz=640,

    batch=4,

    workers=0,

    plots=True
)


# ---------------------------------------------------------
# MÉTRICAS
# ---------------------------------------------------------

print()
print("=" * 60)
print("RESULTADOS")
print("=" * 60)


try:

    precision = metricas.box.mp

    recall = metricas.box.mr

    map50 = metricas.box.map50

    map5095 = metricas.box.map


    print()
    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"mAP50: {map50:.4f}"
    )

    print(
        f"mAP50-95: {map5095:.4f}"
    )


except Exception as error:

    print()
    print(
        "Las métricas fueron calculadas, "
        "pero no fue posible mostrarlas automáticamente."
    )

    print(error)


print()
print("=" * 60)
print("EVALUACIÓN FINALIZADA")
print("=" * 60)