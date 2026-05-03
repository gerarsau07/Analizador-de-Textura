# Analizador de Textura 

Este repositorio contiene un sistema de **Machine Learning** desarrollado en Python para la clasificación automatizada de texturas. El proyecto implementa un flujo de trabajo profesional que abarca desde el preprocesamiento de datos hasta la validación de modelos de alta precisión mediante **XGBoost**.

## Descripción del Proyecto

El sistema toma datos extraídos de imágenes (características numéricas de texturas) y utiliza técnicas de aprendizaje supervisado para identificar patrones distintivos. Se enfoca en la robustez estadística mediante validación cruzada y la optimización de hiperparámetros para garantizar resultados confiables en entornos de producción.

##  Funcionalidades Principales

*   **Procesamiento de Datos:** Limpieza y normalización de características utilizando `StandardScaler`.
*   **Validación Cruzada Estratificada (5-Fold):** Evaluación del modelo en múltiples subconjuntos de datos para asegurar su capacidad de generalización.
*   **Modelo de Alto Rendimiento:** Implementación de `XGBClassifier` con optimización de árboles y parada temprana (*early stopping*) para evitar el sobreajuste.
*   **Métricas de Evaluación:** Reportes detallados de Accuracy, F1-Score (Macro) y curvas AUC-ROC.
*   **Exportación del Modelo:** Generación de archivos binarios para el modelo (`.xgb`) y el escalador (`.pkl`) listos para inferencia.
*   **Visualización Técnica:** Generación automática de gráficas de importancia de características y curvas de pérdida (Log-Loss).

## Stack Tecnológico

*   **Lenguaje:** Python 3.12+
*   **Librerías de Ciencia de Datos:** `Pandas`, `NumPy`
*   **Machine Learning:** `Scikit-learn`, `XGBoost`
*   **Visualización:** `Matplotlib`
*   **Gestión de Archivos:** `Joblib`, `OS`

## Configuración del Modelo

El entrenamiento final se realiza con una configuración de ensamble optimizada:
*   **Máximo de Estimadores:** 1000
*   **Learning Rate:** 0.05
*   **Profundidad Máxima:** 5
*   **Estrategia de Regularización:** Subsampling (0.8) y Colsample_bytree (0.8).

## Estructura de Salida

Al ejecutar el código, se genera el directorio `modelo_texturas/` con los siguientes recursos:

| Archivo | Descripción |
| :--- | :--- |
| `modelo_final.xgb` | Modelo entrenado en formato binario. |
| `scaler.pkl` | Transformador de datos normalizado. |
| `importancia_caracteristicas.png` | Gráfico de las variables con mayor impacto en la predicción. |
| `curva_aprendizaje.png` | Análisis visual del progreso de entrenamiento. |

