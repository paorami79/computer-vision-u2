import sys


def verificar_libreria(nombre, nombre_import):
    try:
        modulo = __import__(nombre_import)
        version = getattr(modulo, "__version__", "versión no disponible")

        print(f"[OK] {nombre}: {version}")
        return True

    except ImportError:
        print(f"[ERROR] {nombre}: NO está instalada")
        return False


print("=" * 60)
print("UNIDAD 2 - YOLOv8")
print("VERIFICACIÓN DEL ENTORNO")
print("=" * 60)

print()
print("Versión de Python:")
print(sys.version)
print()

librerias = [
    ("NumPy", "numpy"),
    ("OpenCV", "cv2"),
    ("Albumentations", "albumentations"),
    ("PyYAML", "yaml"),
    ("Requests", "requests"),
    ("Matplotlib", "matplotlib"),
    ("Pandas", "pandas"),
    ("Ultralytics", "ultralytics"),
]

total = len(librerias)
correctas = 0

for nombre, nombre_import in librerias:

    if verificar_libreria(nombre, nombre_import):
        correctas += 1


print()
print("=" * 60)
print(f"RESULTADO: {correctas}/{total} librerías disponibles")
print("=" * 60)

if correctas == total:

    print()
    print("El entorno está preparado correctamente.")
    print("Puede continuar con el Paso 2.")

else:

    print()
    print("Hay librerías pendientes de instalación.")
    print()
    print("Ejecute:")
    print()
    print("pip install -r requirements.txt")
