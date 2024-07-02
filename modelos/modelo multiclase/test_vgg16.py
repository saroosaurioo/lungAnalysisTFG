import itertools
import numpy as np
import os
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import colors
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import csv

# Definición de variables
data_dir = r'..\..\..\dataset'
categorias = ['No_findings','Covid-19','Pneumonia']
category_indices = {category: i for i, category in enumerate(categorias)}

# Cargar y preprocesar imágenes
def cargar_preprocesar_imagenes(data_dir, category_indices):
    imagenes = []
    labels = []
    for category, label in category_indices.items():
        path = os.path.join(data_dir, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            imagenes.append(image)
            labels.append(label)
    return np.array(imagenes), np.array(labels)

imagenes, labels = cargar_preprocesar_imagenes(data_dir, category_indices)
labels = to_categorical(labels, num_classes=len(categorias))

# Dividir las imágenes
_, X_val, _, y_val = train_test_split(imagenes, labels, test_size=0.2, stratify=labels, random_state=42)

# Extracción de características con VGG16
model_vgg16 = VGG16(weights='imagenet', include_top=False)
features_val = model_vgg16.predict(X_val)

def evaluar_modelo(nombre_modelo, features_val, y_val):
    modelo_clasificador = load_model(nombre_modelo)
    scores = modelo_clasificador.evaluate(features_val, y_val)
    print(f"Loss en validación para {nombre_modelo}: {scores[0]}")
    print(f"Accuracy en validación para {nombre_modelo}: {scores[1]}")

    # Predecir las etiquetas
    predicciones = modelo_clasificador.predict(features_val)
    predicciones_etiquetas = np.argmax(predicciones, axis=1)
    y_val_etiquetas = np.argmax(y_val, axis=1)

    predictions = predicciones.argmax(axis=1)


    y_val_csv = y_val.argmax(axis=1)
    # Guardar valores en un CSV
    with open('resultados.csv', 'w', newline='') as csvfile:
        fieldnames = ['Valor Verdadero', 'Prediccion']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for true_label, pred_label in zip(y_val[:, 1], predicciones[:, 1]):
            writer.writerow({'Valor Verdadero': true_label, 'Prediccion': pred_label})

    # Matriz de confusión
    plt.figure(figsize=(10,8))
    matriz_confusion = confusion_matrix(y_val_etiquetas, predicciones_etiquetas)
    plt.imshow(matriz_confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(categorias))
    plt.xticks(tick_marks, categorias, rotation=0)


    plt.yticks(tick_marks, categorias)
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Verdaderos')
    thresh = matriz_confusion.max() / 2.
    for i, j in itertools.product(range(matriz_confusion.shape[0]), range(matriz_confusion.shape[1])):
        plt.text(j, i, matriz_confusion[i, j], horizontalalignment="center", color="white" if matriz_confusion[i, j] > matriz_confusion.max() / 2. else "black")
    plt.show()


modelos = ['modelo3_vgg16.h5']

# Evaluar todos los modelos
for modelo in modelos:
    evaluar_modelo(modelo, features_val, y_val)