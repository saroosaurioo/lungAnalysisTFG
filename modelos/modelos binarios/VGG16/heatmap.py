import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras import backend as K
import tensorflow as tf

# Cargar el modelo VGG16 preentrenado
model = VGG16(weights='imagenet')


# Función para obtener el heatmap
def generate_heatmap(img_path, model, last_conv_layer_name, pred_index=None):
    # Cargar y preprocesar la imagen
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Crear un modelo que mapea la imagen de entrada a las activaciones del último conv layer
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Obtener el índice de la clase predicha

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = np.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Obtener los gradientes de la clase con respecto a la salida del último conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # Obtener los gradientes medios de cada filtro
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # Pesar la salida del conv layer por los gradientes medios
    conv_outputs = conv_outputs[0]
    heatmap = np.dot(conv_outputs, pooled_grads[..., np.newaxis])[..., 0]

    # Aplicar ReLU a la heatmap
    heatmap = np.maximum(heatmap, 0)

    # Normalizar la heatmap
    heatmap /= np.max(heatmap)

    return heatmap


# Función para superponer el heatmap en la imagen original
def superimpose_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # Cargar la imagen original
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Error al cargar la imagen desde la ruta: {img_path}")

    # Redimensionar el heatmap al tamaño de la imagen original
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convertir el heatmap a RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Superponer el heatmap en la imagen original
    superimposed_img = heatmap * alpha + img

    return superimposed_img


# Path de la imagen para probar, añadir el path completo con el nombre de la imagen que se quiere probar
img_path =  r'..\..\..\dataset\No_findings\00001281_000.png'

# Generar el heatmap
heatmap = generate_heatmap(img_path, model, last_conv_layer_name='block5_conv3')

# Superponer el heatmap en la imagen original
superimposed_img = superimpose_heatmap(img_path, heatmap)

# Mostrar la imagen con el heatmap superpuesto
cv2.imwrite('C:/Users/sarit/OneDrive/Escritorio/heatmap.png', superimposed_img)