import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
import os
import pandas as pd

def procesar_imagen(img_path, nlevels=256):
    I = cv2.imread(img_path)
    if I is None:
        print(f"No se pudo leer la imagen: {img_path}")
        return None
    
    img_name = os.path.basename(img_path)
    resultados = []
    
    for angle in [0, 45, 90, 135]:
        if angle != 0:
            rows, cols = I.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            I_rot = cv2.warpAffine(I, M, (cols, rows))
        else:
            I_rot = I.copy()
        
        # Convertir a escala de grises
        imgG = cv2.cvtColor(I_rot, cv2.COLOR_BGR2GRAY)
        pek_uint8 = np.uint8(imgG)
        
        # Calcular GLCM
        glcm = graycomatrix(pek_uint8,
                          distances=[1],
                          angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                          levels=nlevels,
                          symmetric=True,
                          normed=True)
        
        # Calcular propiedades de textura
        stats_homg = graycoprops(glcm, 'homogeneity')
        stats_contrast = graycoprops(glcm, 'contrast')
        stats_energy = graycoprops(glcm, 'energy')
        stats_corr = graycoprops(glcm, 'correlation')
        stats_dissimilarity = graycoprops(glcm, 'dissimilarity')
        stats_asm = graycoprops(glcm, 'ASM')
        
        # Promedio por ángulo
        homg = np.nanmean(stats_homg, axis=1)[0]
        contrast = np.nanmean(stats_contrast, axis=1)[0]
        energy = np.nanmean(stats_energy, axis=1)[0]
        corr = np.nanmean(stats_corr, axis=1)[0]
        dissimilarity = np.nanmean(stats_dissimilarity, axis=1)[0]
        asm = np.nanmean(stats_asm, axis=1)[0]
        
        # Guardar resultados
        resultados.append({
            'Imagen': img_name,
            'Angulo_Rotacion': angle,
            'Homogeneidad': homg,
            'Contraste': contrast,
            'Energia': energy,
            'Correlacion': corr,
            'Disimilaridad': dissimilarity,
            'ASM': asm
        })
    
    return resultados

def procesar_carpeta(carpeta, output_csv='resultados_texturafin.csv'):
    
    formatos = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
    imagenes = [os.path.join(carpeta, f) for f in os.listdir(carpeta) 
               if f.lower().endswith(formatos)]
    
    if not imagenes:
        print("No se encontraron imágenes en la carpeta.")
        return
    

    todos_resultados = []
    print("\nProcesando imágenes:")
    for i, img_path in enumerate(imagenes, 1):
        print(f"  {i}/{len(imagenes)}: {os.path.basename(img_path)}")
        res = procesar_imagen(img_path)
        if res:
            todos_resultados.extend(res)
    
    # Crear DataFrame 
    df = pd.DataFrame(todos_resultados)
    df = df.sort_values(by=['Imagen', 'Angulo_Rotacion'])
    df.to_csv(output_csv, index=False)
    print(f"\nResultados guardados en: {output_csv}")
    
    print("\nResumen de resultados:")
    print(df.groupby('Imagen').mean(numeric_only=True))
    
    return df


if __name__ == "__main__":
    carpeta_imagenes = ''
    resultados = procesar_carpeta(carpeta_imagenes)