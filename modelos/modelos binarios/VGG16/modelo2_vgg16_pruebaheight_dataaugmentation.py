import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16

# Ruta al conjunto de datos
data_path = r'..\..\..\dataset'

# Cargar imágenes y etiquetas
imagenes = []
labels = []

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#Cargar y preprocesar las imagenes
for categoria in ['No_findings', 'Covid-19', 'Pneumonia']:
    dir_path = os.path.join(data_path, categoria)
    for img in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img)
        imagen = Image.open(img_path).resize((224, 224)).convert('RGB')
        imagenes.append(np.array(imagen))
        labels.append(categoria)

# Mapeo de etiquetas a clases binarias
nueva_etiqueta = {'No_findings': 0, 'Covid-19': 1, 'Pneumonia': 1}
labels = [nueva_etiqueta[label] for label in labels]

# Convertir listas a arrays de NumPy
imagenes = np.array(imagenes)
labels = np.array(labels)

labels = to_categorical(labels, num_classes=2)

# División en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(imagenes, labels, test_size=0.2, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Crear generadores de imágenes con Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    height_shift_range=0.2
)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
validation_generator = validation_datagen.flow(X_val, y_val, batch_size=32)

# Crear el modelo usando la arquitectura VGG16
vgg_base = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
model = Sequential(vgg_base.layers)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])



# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=len(X_val) // 32,
    callbacks=[early_stopping]
)

# Guardar el modelo entrenado
model.save("modelo2_vgg16_height_dataaugmentationg.h5")

# Evaluación del modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Accuracy en el conjunto de prueba:', test_acc)

# Gráficas de loss y accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
