import cv2
import os
import numpy as np
from tqdm import tqdm # Si no lo tienes, usa: pip install tqdm

# 1. Configuración de rutas77
ruta_origen = r"C:/Users/gerar/Desktop/Analizador-de-Textura-main/Modelo2026/10clases"
ruta_destino = r"C:/Users/gerar/Desktop/Analizador-de-Textura-main/Modelo2026/10clasesAumentado"

def procesar_dataset(origen, destino):
    if not os.path.exists(destino):
        os.makedirs(destino)

    clases = os.listdir(origen)
    
    for clase in clases:
        path_clase_orig = os.path.join(origen, clase)
        path_clase_dest = os.path.join(destino, clase)
        
        # Saltamos si no es una carpeta
        if not os.path.isdir(path_clase_orig):
            continue
            
        os.makedirs(path_clase_dest, exist_ok=True)
        print(f"Procesando clase: {clase}...")

        imagenes = [f for f in os.listdir(path_clase_orig) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in tqdm(imagenes):
            img_path = os.path.join(path_clase_orig, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue

            # --- A. Guardar Original ---
            cv2.imwrite(os.path.join(path_clase_dest, f"original_{img_name}"), img)

            # --- B. Rotación 45 grados (con relleno de textura) ---
            (h, w) = img.shape[:2]
            centro = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(centro, 45, 1.0)
            # BORDER_REFLECT evita esquinas negras reflejando la misma imagen
            img_45 = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            cv2.imwrite(os.path.join(path_clase_dest, f"rot45_{img_name}"), img_45)

            # --- C. Rotación 90 grados ---
            img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(path_clase_dest, f"rot90_{img_name}"), img_90)

    print(f"\n✅ ¡Listo! Dataset aumentado guardado en: {destino}")

# Ejecutar la función
procesar_dataset(ruta_origen, ruta_destino)