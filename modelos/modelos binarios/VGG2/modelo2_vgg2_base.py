import os
import numpy as np
from PIL import Image
from keras import Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt



data_path = r'..\..\..\dataset'
categorias = ['Sano','No Sano']

nueva_etiqueta = {'No_findings': 0, 'Covid-19': 1, 'Pneumonia': 1}

imagenes = []
labels = []

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#Recorrer categorias definidas, cargar cada imagen de cada categoria, prepocesarla y redimensionarla, agregar a la lista de imagenes y las etiquetas a la lista de etiquetas
for i, category in enumerate(nueva_etiqueta):
    path = os.path.join(data_path, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        imagenes.append(image)
        labels.append(nueva_etiqueta[category])

#Convertir la lista de imagenes y etiquetas a arrays
imagenes = np.array(imagenes)
labels = np.array(labels)

labels = to_categorical(labels, num_classes=len(categorias))

#Dividir las imagenes en entrenamiento(80%) y eso(20%)
X_train, X_temp, y_train, y_temp = train_test_split(imagenes, labels, test_size=0.2, stratify=labels, random_state=42)
#Division validacion y prueba (50/50)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,stratify=y_temp, random_state=42)

#Definicion de la arquitectura VGG2
def model_VGG_2(nRows, nCols, depth, nFilters):
    visible = Input(shape=(nRows, nCols, depth))
    layer = vgg_block(visible, nFilters, 2)
    layer = vgg_block(layer, 2 * nFilters, 2)

    layer = Flatten()(layer)
    layer = Dense(128, activation='relu', kernel_initializer='he_uniform')(layer)
    layer = Dense(2, activation='sigmoid')(layer)

    model = Model(inputs=visible, outputs=layer)

    return model


def vgg_block(layer_in, n_filters, n_conv):
    for _ in range(n_conv):
        layer_in = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(layer_in)
    layer_in = MaxPooling2D((2, 2), strides=(2, 2))(layer_in)

    return layer_in

model = model_VGG_2(nRows=224, nCols=224, depth=3, nFilters=32)
model.summary()
# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

epochs_prueba = 100

# Entrenar el modelo
historial = model.fit(X_train, y_train, epochs = epochs_prueba, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])



model.save("modelo2_vgg2_base.h5")

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy*100:.2f}%")


# Graficar la pérdida
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento y la validación')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend()
plt.show()

# Graficar la precisión
plt.plot(historial.history['accuracy'], label='Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento y la validación')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend()
plt.show()