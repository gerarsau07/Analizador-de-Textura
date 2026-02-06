# Analizador de Texturas con IA

Este proyecto es un sistema de clasificación de imágenes basado en aprendizaje automático capaz de identificar tres tipos específicos de texturas: **Burbuja, Cristal y Waffle**. Incluye un pipeline completo desde el procesamiento de imágenes y creación de datasets hasta el entrenamiento del modelo y una interfaz gráfica de usuario (GUI).

## Características
* **Clasificación Tri-clase:** Identifica texturas de burbuja, cristal y waffle.
* **Extracción de Características:** Generación automática de datasets a partir de imágenes (`CrearCSV.py`).
* **Modelo Entrenado:** Incluye un modelo listo para usar almacenado en formato `.joblib`.
* **Interfaz Gráfica:** GUI intuitiva desarrollada en Python para facilitar el uso sin necesidad de terminal.

## Estructura del Proyecto
* `CrearCSV.py`: Script para procesar las imágenes y extraer las características necesarias para el entrenamiento.
* `EntrenamientoModelo.py`: Código para entrenar el clasificador y evaluar su desempeño.
* `InterfazGrafica.py`: Aplicación principal con la que el usuario interactúa.
* `texture_classifier_model.joblib`: El cerebro del proyecto (modelo exportado).
* `resultados_textura.csv`: Datasets generados durante el proceso de extracción.

## Tecnologías Utilizadas
* **Lenguaje:** Python 3.x
* **IA/ML:** Scikit-learn, Joblib.
* **Procesamiento de Datos:** Pandas, NumPy.
* **Interfaz:** Tkinter / PySimpleGUI (según tu implementación en el archivo).

## 📦 Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/gerarsau07/Analizador-de-Textura.git](https://github.com/gerarsau07/Analizador-de-Textura.git)
   cd Analizador-de-Textura
