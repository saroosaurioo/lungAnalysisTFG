import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

data_path = r'..\..\..\dataset'
model_paths = [
    "modelo2_vgg16_base.h5",
    "modelo2_vgg16_base_dataaugmentation.h5",
    "modelo2_vgg16_base_dropout.h5",
    "modelo2_vgg16_transferLearning.h5",
    "modelo2_vgg16_fineTuning.h5"
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


def evaluar_modelos(model_paths, X_test, y_test, X_test_features):
    resultados = {}
    roc_data = []  # Para almacenar los datos de la curva ROC de cada modelo
    combined_data = []  # Para almacenar los datos combinados de y_test y predictions

    for idx, path in enumerate(model_paths):
        model = load_model(path)
        try:
            if 'transferLearning' in path or 'fineTuning' in path:
                predictions = model.predict(X_test_features)
                test_loss, test_acc = model.evaluate(X_test_features, y_test, verbose=0)
            else:
                predictions = model.predict(X_test)
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            predictions_lista = predictions.flatten().tolist()
            y_test_lista = y_test.flatten().tolist()
            resultados[path] = test_acc
            print(f"Modelo: {path}, Accuracy en test: {test_acc:.3f}")
            # Almacenar datos de y_test y predictions en una lista
            combined_data.append({
                'Modelo': path,
                'y_test': y_test[:, 1],
                'predictions': predictions[:, 1]
            })

            # Generar y mostrar la matriz de confusión en una nueva figura
            cm = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
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

    # Combina todos los datos de y_test y predictions en un solo DataFrame
    df_combined = pd.DataFrame(combined_data)

    # Exportar todos los datos combinados a un archivo CSV
    df_combined.to_csv("y_test_predictions_combined.csv", index=False, sep="|")

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
