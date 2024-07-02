from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

# Cargar el modelo clasificador que ya entrenaste y guardaste
modelo_clasificador = load_model('modelo3_vgg16.h5')

# Ruta de la imagen de prueba, cambiar la ruta a la de la imagen seleccionada
imagen_prueba_path = r'..\..\..\dataset\No_findings\00001281_000.png'

# Cargar la imagen con el tamaño adecuado
imagen_prueba = load_img(imagen_prueba_path, target_size=(224, 224))

# Convertir la imagen a un array
imagen_prueba_array = img_to_array(imagen_prueba)

# Preprocesar la imagen (como lo hiciste para el entrenamiento)
imagen_prueba_array = np.expand_dims(imagen_prueba_array, axis=0)
imagen_prueba_array = preprocess_input(imagen_prueba_array)

# Usar VGG16 para extraer características, como hiciste en entrenamiento
model_vgg16 = VGG16(weights='imagenet', include_top=False)
features_imagen_prueba = model_vgg16.predict(imagen_prueba_array)

# Predecir la clase de la imagen utilizando tu modelo clasificador
prediccion = modelo_clasificador.predict(features_imagen_prueba)

# Decodificar la prediccion (para obtener la etiqueta de la clase)
etiqueta_predicha = np.argmax(prediccion, axis=1)
categorias = ['Normal','Covid-19','Pneumonia']
print(f"La imagen es: {categorias[etiqueta_predicha[0]]}")