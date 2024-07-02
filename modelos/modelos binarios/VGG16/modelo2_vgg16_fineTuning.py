import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping



data_dir = r'..\..\..\dataset'

#Catergorías a clasificar
categorias = ['Sano','No Sano']

nueva_etiqueta = {'No_findings': 0, 'Covid-19': 1, 'Pneumonia': 1}

#Modelo vgg16 preentrenado, excluyendo las capas superiores
model = VGG16(weights='imagenet', include_top=False)
for layer in model.layers[:-3]:
    layer.trainable = False

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

imagenes = []
labels = []

#Recorrer categorias definidas, cargar cada imagen de cada categoria, prepocesarla y redimensionarla, agregar a la lista de imagenes y las etiquetas a la lista de etiquetas
for i, category in enumerate(nueva_etiqueta):
    path = os.path.join(data_dir, category)
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


print("Entrenamiento:", X_train.shape[0])
print("Validacion:", X_val.shape[0])
print("Pruebas:", X_test.shape[0])


#Extracción de las caracteristicas de cada conjunto de datos
features_train = model.predict(X_train)
features_val = model.predict(X_val)
features_test = model.predict(X_test)

#configuración del imagedatagenerator para data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #shear_range=0.15,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


#Modelo secuencial para la clasificacion de las características
modelo_clasificador = Sequential([
    #Convertir entradas multidimensionales en un vector unidimensional
    Flatten(input_shape = (7,7,512)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    #capa de salida
    Dense(2, activation='sigmoid')
])


modelo_clasificador.compile(loss='binary_crossentropy',metrics=['accuracy'])

modelo_clasificador.summary()
train_generator = datagen.flow(features_train, y_train, batch_size=8)

epochs_prueba = 100

print("Entrenamiento2:", X_train.shape[0])
print("Validacion2:", X_val.shape[0])
print("Pruebas2:", X_test.shape[0])

#Entrenar el modelo clasificador
historial = modelo_clasificador.fit(train_generator,
                                    validation_data=(features_val,y_val),
                                    epochs=epochs_prueba,
                                    batch_size=8,
                                    callbacks=[early_stopping])

modelo_clasificador.save('modelo2_vgg16_fineTuning.h5')

modelo_clasificador.summary()

#Evaluacion del modelo
scores = modelo_clasificador.evaluate(features_test, y_test)
print("Loss: ", scores[0])
print("Accuracy: ", scores[1])


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