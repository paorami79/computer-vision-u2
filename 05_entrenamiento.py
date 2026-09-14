from ultralytics import YOLO
from pathlib import Path

# ============================================================
# CONFIGURACIÓN DEL PROYECTO
# ============================================================

BASE_DIR = Path(
    r"C:\Users\GreenSQA\OneDrive - GreenSQA\Personales\Estudio especializacion\Python\Proyecto_YOLOv8_Unidad2"
)

DATA_YAML = BASE_DIR / "data.yaml"

TRAIN_IMAGES = BASE_DIR / "dataset" / "images" / "train"
VAL_IMAGES = BASE_DIR / "dataset" / "images" / "val"

MODELOS = BASE_DIR / "modelos"

# ============================================================
# VERIFICACIÓN
# ============================================================

print("=" * 70)
print("YOLOv8 - ENTRENAMIENTO")
print("=" * 70)

print("\nArchivo data.yaml:")
print(DATA_YAML)

print("\nTrain:")
print(TRAIN_IMAGES)

print("\nVal:")
print(VAL_IMAGES)

# Comprobar que existen las carpetas
if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"\nERROR: No existe data.yaml:\n{DATA_YAML}"
    )

if not TRAIN_IMAGES.exists():
    raise FileNotFoundError(
        f"\nERROR: No existe la carpeta TRAIN:\n{TRAIN_IMAGES}"
    )

if not VAL_IMAGES.exists():
    raise FileNotFoundError(
        f"\nERROR: No existe la carpeta VAL:\n{VAL_IMAGES}"
    )

# Contar imágenes
train_count = len(list(TRAIN_IMAGES.glob("*.jpg")))
val_count = len(list(VAL_IMAGES.glob("*.jpg")))

print("\n" + "-" * 70)
print("IMÁGENES ENCONTRADAS")
print("-" * 70)

print(f"TRAIN: {train_count}")
print(f"VAL:   {val_count}")

if train_count == 0:
    raise RuntimeError("ERROR: TRAIN no tiene imágenes.")

if val_count == 0:
    raise RuntimeError("ERROR: VAL no tiene imágenes.")

print("\nDataset encontrado correctamente.")

# ============================================================
# CARGAR YOLOv8
# ============================================================

print("\nCargando YOLOv8n...")

model = YOLO("yolov8n.pt")

# ============================================================
# ENTRENAMIENTO
# ============================================================

print("\nIniciando entrenamiento...")
print("Puede tardar bastante porque se está utilizando CPU.")

model.train(
    data=str(DATA_YAML.resolve()),
    epochs=1,
    batch=4,
    imgsz=320,
    workers=0,
    device="cpu",
    fraction=0.01,
    project=str(MODELOS.resolve()),
    name="yolov8_coco",
    exist_ok=True
)

print("\n" + "=" * 70)
print("ENTRENAMIENTO TERMINADO")
print("=" * 70)

print("\nLos resultados están en:")

print(
    MODELOS / "yolov8_coco"
)

print("\nModelo entrenado:")
print(
    MODELOS / "yolov8_coco" / "weights" / "best.pt"
)