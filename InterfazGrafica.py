import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import joblib
import os
import cv2
from skimage.feature import graycomatrix, graycoprops

class TextureClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Texturas")
        self.root.geometry("1400x900")  
        
        self.aciertos = 0
        self.errores = 0
        
        self.real_class_mapping = {
            'bubbly': 0,
            'crystalline': 1,
            'waffled': 2
        }
        
        self.example_images = {
            0: "ejemplos_clases/bubbly_0069.jpg",
            1: "ejemplos_clases/crystalline_0104.jpg",
            2: "ejemplos_clases/waffled_0075.jpg"
        }
        
        # Cargar el modelo entrenado
        try:
            self.model = joblib.load('texture_classifier_model.joblib')
            self.feature_names = ['Homogeneidad', 'Contraste', 'Energia', 'Correlacion', 'Disimilaridad', 'ASM']
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {str(e)}")
            self.root.destroy()
            return
        
        # Configurar la interfaz
        self.setup_ui()
        
    def setup_ui(self):
        # Frame principal
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame para estadísticas
        stats_frame = tk.Frame(main_frame)
        stats_frame.pack(fill=tk.X, pady=5)
        
        # Labels para contadores
        tk.Label(stats_frame, text="Rendimiento del modelo:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.aciertos_label = tk.Label(stats_frame, text="Aciertos: 0", fg="green")
        self.aciertos_label.pack(side=tk.LEFT, padx=10)
        self.errores_label = tk.Label(stats_frame, text="Errores: 0", fg="red")
        self.errores_label.pack(side=tk.LEFT, padx=10)
        self.precision_label = tk.Label(stats_frame, text="Precisión: 0.00%")
        self.precision_label.pack(side=tk.LEFT, padx=10)
        
        # Frame para imágenes (original, predicha, real)
        self.images_frame = tk.Frame(main_frame)
        self.images_frame.pack(pady=10)
        
        # Frame para la imagen cargada
        self.original_frame = tk.Frame(self.images_frame, width=300, height=300, bg='white', 
                                   relief=tk.SUNKEN, borderwidth=1)
        self.original_frame.pack_propagate(False)
        self.original_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        tk.Label(self.images_frame, text="Imagen Cargada", font=("Arial", 9, "bold")).grid(row=1, column=0)
        self.original_image_label = tk.Label(self.original_frame)
        self.original_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Frame para la imagen de la clase predicha
        self.predicted_frame = tk.Frame(self.images_frame, width=300, height=300, bg='white', 
                                   relief=tk.SUNKEN, borderwidth=1)
        self.predicted_frame.pack_propagate(False)
        self.predicted_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        tk.Label(self.images_frame, text="Clase Predicha", font=("Arial", 9, "bold")).grid(row=1, column=1)
        self.predicted_image_label = tk.Label(self.predicted_frame)
        self.predicted_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Frame para la imagen de la clase real
        self.real_frame = tk.Frame(self.images_frame, width=300, height=300, bg='white', 
                                   relief=tk.SUNKEN, borderwidth=1)
        self.real_frame.pack_propagate(False)
        self.real_frame.grid(row=0, column=2, padx=10, pady=5, sticky="nsew")
        tk.Label(self.images_frame, text="Clase Real", font=("Arial", 9, "bold")).grid(row=1, column=2)
        self.real_image_label = tk.Label(self.real_frame)
        self.real_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Botón para cargar imagen
        load_btn = tk.Button(main_frame, text="Cargar Imagen", 
                            command=self.load_image, width=20)
        load_btn.pack(pady=5)
        
        # Botón para clasificar
        classify_btn = tk.Button(main_frame, text="Clasificar Textura", 
                               command=self.classify_texture, width=20)
        classify_btn.pack(pady=5)
        
        # Área de resultados
        self.result_text = tk.Text(main_frame, height=15, width=100, state=tk.DISABLED)
        self.result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Diccionario de clases
        self.class_names = {
            0: "Burbujas",
            1: "Cristalino",
            2: "Waffle"
        }
        
    def display_image(self, image_path, label_widget, size=(300, 300)):
        """Muestra una imagen en el widget especificado"""
        try:
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path)
                image.thumbnail(size)
                photo = ImageTk.PhotoImage(image)
                label_widget.config(image=photo)
                label_widget.image = photo
            else:
                label_widget.config(image='', text="Imagen no disponible", font=("Arial", 10))
                label_widget.image = None
        except Exception as e:
            label_widget.config(image='', text=f"Error: {str(e)}", font=("Arial", 9))
            label_widget.image = None
        
    def actualizar_estadisticas(self):
        """Actualiza las etiquetas de estadísticas en la interfaz"""
        total = self.aciertos + self.errores
        precision = (self.aciertos / total * 100) if total > 0 else 0
        
        self.aciertos_label.config(text=f"Aciertos: {self.aciertos}")
        self.errores_label.config(text=f"Errores: {self.errores}")
        self.precision_label.config(text=f"Precisión: {precision:.2f}%")
        
    def determinar_clase_real(self, filename):
        """Determina la clase real basada en el nombre del archivo"""
        for prefix, class_id in self.real_class_mapping.items():
            if filename.lower().startswith(prefix):
                return class_id
        return None  # No se encontró coincidencia
    
    def verificar_prediccion(self, predicted_class, real_class):
        """Verifica si la predicción es correcta y actualiza contadores"""
        if real_class is None:
            return "⚠️ No se pudo determinar la clase real (nombre no reconocido)"
        
        if predicted_class == real_class:
            self.aciertos += 1
            return "✅ ACIERTO (Predicción correcta)"
        else:
            self.errores += 1
            real_name = self.class_names.get(real_class, f"Clase {real_class}")
            return f"❌ ERROR (Clase real: {real_name})"
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos los archivos", "*.*")]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.current_filename = os.path.basename(file_path)
                
                # Mostrar la imagen cargada
                self.display_image(file_path, self.original_image_label)
                
                # Limpiar imágenes de resultados anteriores
                self.predicted_image_label.config(image='')
                self.predicted_image_label.image = None
                self.real_image_label.config(image='')
                self.real_image_label.image = None
                
                # Limpiar resultados anteriores
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Imagen cargada: {self.current_filename}")
                self.result_text.config(state=tk.DISABLED)
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
    
    def extract_texture_features(self, image_path):
        # Cargar imagen en escala de grises
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Redimensionar si es necesario
        img = cv2.resize(img, (256, 256))
        
        # Calcular GLCM
        glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Extraer características de textura
        features = {
            'Homogeneidad': graycoprops(glcm, 'homogeneity')[0, 0],
            'Contraste': graycoprops(glcm, 'contrast')[0, 0],
            'Energia': graycoprops(glcm, 'energy')[0, 0],
            'Correlacion': graycoprops(glcm, 'correlation')[0, 0],
            'Disimilaridad': graycoprops(glcm, 'dissimilarity')[0, 0],
            'ASM': graycoprops(glcm, 'ASM')[0, 0]
        }
        
        return features
    
    def classify_texture(self):
        if not hasattr(self, 'current_image_path'):
            messagebox.showwarning("Advertencia", "Por favor cargue una imagen primero")
            return
            
        try:
            # Extraer características
            features = self.extract_texture_features(self.current_image_path)
            
            # Convertir a formato adecuado para el modelo
            feature_values = [features[name] for name in self.feature_names]
            features_df = pd.DataFrame([feature_values], columns=self.feature_names)
            
            # Realizar predicción
            prediction = self.model.predict(features_df)
            proba = self.model.predict_proba(features_df)
            
            # Determinar clase predicha
            predicted_class = prediction[0]
            predicted_name = self.class_names.get(predicted_class, f"Clase {predicted_class}")
            
            # Determinar clase real basada en el nombre del archivo
            real_class = self.determinar_clase_real(self.current_filename)
            real_name = self.class_names.get(real_class, f"Clase {real_class}") if real_class is not None else "Desconocida"
            
            # Verificar si la predicción es correcta
            verification_result = self.verificar_prediccion(predicted_class, real_class)
            self.actualizar_estadisticas()
            
            # Mostrar imagen de clase predicha
            self.display_image(
                self.example_images.get(predicted_class, ""),
                self.predicted_image_label
            )
            
            # Mostrar imagen de clase real (si se pudo determinar)
            if real_class is not None:
                self.display_image(
                    self.example_images.get(real_class, ""),
                    self.real_image_label
                )
            else:
                self.real_image_label.config(image='', text="Clase real desconocida", font=("Arial", 10))
                self.real_image_label.image = None
            
            # Mostrar resultados
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            
            self.result_text.insert(tk.END, "RESULTADOS DE CLASIFICACIÓN:\n", "title")
            self.result_text.insert(tk.END, f"• Imagen: {self.current_filename}\n")
            self.result_text.insert(tk.END, f"• Clase predicha: {predicted_name}\n")
            self.result_text.insert(tk.END, f"• Clase real: {real_name}\n")
            self.result_text.insert(tk.END, f"• Verificación: {verification_result}\n\n")
            
            self.result_text.insert(tk.END, "PROBABILIDADES POR CLASE:\n", "title")
            for i, prob in enumerate(proba[0]):
                name = self.class_names.get(i, f"Clase {i}")
                self.result_text.insert(tk.END, f"• {name}: {prob*100:.2f}%\n")
            
            self.result_text.insert(tk.END, "\nCARACTERÍSTICAS EXTRAÍDAS:\n", "title")
            for name, value in features.items():
                self.result_text.insert(tk.END, f"• {name}: {value:.4f}\n")
            
            # Configurar estilos para el texto
            self.result_text.tag_configure("title", font=("Arial", 10, "bold"))
            self.result_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la clasificación: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextureClassifierApp(root)
    root.mainloop()