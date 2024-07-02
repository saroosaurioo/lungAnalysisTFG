import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
# Configuración de rutas y modelos
data_path = r'..\..\..\dataset'
model_paths = [
    "modelo2_vgg16_base_dataaugmentationg.h5",
    "modelo2_vgg16_horizontal_dataaugmentationg.h5",
    "modelo2_vgg16_shear_dataaugmentationg.h5",
    "modelo2_vgg16_fill_dataaugmentationg.h5",
    "modelo2_vgg16_height_dataaugmentationg.h5",
    "modelo2_vgg16_pruebas_dataaugmentationg.h5",
    "modelo2_vgg16_zoom_dataaugmentationg.h5",
    "modelo2_vgg16_width_dataaugmentationg.h5"
]
categorias = ['No_findings', 'Covid-19', 'Pneumonia']
nueva_etiqueta = {'No_findings': 0, 'Covid-19': 1, 'Pneumonia': 1}
input_shape = (224, 224)

# Cargar y preprocesar imágenes y etiquetas
def cargar_datos():
    imagenes = []
    labels = []
    for categoria in categorias:
        dir_path = os.path.join(data_path, categoria)
        for img in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img)
            imagen = Image.open(img_path).resize(input_shape).convert('RGB')
            imagenes.append(np.array(imagen))
            labels.append(nueva_etiqueta[categoria])
    imagenes = preprocess_input(np.array(imagenes))
    labels = to_categorical(np.array(labels), num_classes=2)
    return imagenes, labels

# Cargar datos
imagenes, labels = cargar_datos()
X_train, X_temp, y_train, y_temp = train_test_split(imagenes, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Cargar el modelo base de VGG16 para extracción de características
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features(samples):
    return base_model.predict(samples)

# Extraer características para modelos que las necesitan
X_test_features = extract_features(X_test)

# Función para evaluar modelos , generar matrices de confusión y curvas ROC
def evaluar_modelos(model_paths, X_test, y_test, X_test_features):
    resultados = {}
    roc_data = []  # Para almacenar los datos de la curva ROC de cada modelo

    for idx, path in enumerate(model_paths):
        model = load_model(path)
        try:
            if 'transferLearning' in path or 'fineTuning' in path:
                predictions = model.predict(X_test_features)
                test_loss, test_acc = model.evaluate(X_test_features, y_test, verbose=0)
            else:
                predictions = model.predict(X_test)
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

            resultados[path] = test_acc
            print(f"Modelo: {path}, Accuracy en test: {test_acc:.3f}")

            # Convertir probabilidades a etiquetas predichas
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(y_test, axis=1)

            # Generar y mostrar la matriz de confusión en una nueva figura
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
            display_labels = ['San', 'Non San']
            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            display.plot(ax=ax_cm, values_format='d', cmap='Blues')
            ax_cm.set_title(f"Matriz de confusión para {path}")
            plt.show()

            # Almacenar datos de ROC para graficar después
            fpr, tpr, _ = roc_curve(y_test[:, 1], predictions[:, 1])
            roc_auc = auc(fpr, tpr)
            roc_data.append((fpr, tpr, roc_auc, path))

        except Exception as e:
            print(f"Error al evaluar el modelo {path}: {str(e)}")

    # Graficar todas las curvas ROC acumuladas
    plt.figure(figsize=(10, 8))
    for fpr, tpr, roc_auc, label in roc_data:
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC de los diferentes modelos')
    plt.legend(loc="lower right")
    plt.show()

    return resultados

# Evaluar todos los modelos
resultados = evaluar_modelos(model_paths, X_test, y_test, X_test_features)

# Opcional: Graficar los resultados
plt.bar(range(len(resultados)), list(resultados.values()), align='center')
plt.xticks(range(len(resultados)), list(resultados.keys()), rotation=45)
plt.ylabel('Accuracy')
plt.title('Comparación de la precisión de los modelos')
plt.show()
