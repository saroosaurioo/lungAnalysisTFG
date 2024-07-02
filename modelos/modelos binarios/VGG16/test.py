import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm

# Configuración de rutas y modelos
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

# Función para evaluar modelos, generar matrices de confusión y curvas ROC
def evaluar_modelos(model_paths, X_test, y_test, X_test_features):
    resultados = {}
    roc_data = []

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

            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(y_test, axis=1)

            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            display_labels = ['Normal', 'Patologico']
            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            display.plot(ax=ax_cm, values_format='d', cmap='Blues')
            ax_cm.set_title(f"Matriz de confusión para {path}")
            plt.show()

            # KDE Plot para visualizar la distribución de las probabilidades predichas para ambas clases
            plt.figure(figsize=(8, 6))
            # Filtrar las predicciones por clase y trazarlas con diferentes colores
            sns.kdeplot(predictions[y_pred == 0, 1], color='red', bw_adjust=0.5, fill=True, label='Clase 0')
            sns.kdeplot(predictions[y_pred == 1, 1], color='blue', bw_adjust=0.5, fill=True, label='Clase 1')
            plt.title(f'Distribución de probabilidad predicha (KDE) para {path}')
            plt.xlabel('Probabilidad predicha de la clase 1')
            plt.ylabel('Densidad')
            plt.legend(title='Clase Predicha')
            plt.show()

            # QQ Plots para visualizar la normalidad de las distribuciones predichas para ambas clases
            # Clase 0
            plt.figure(figsize=(6, 6))
            qqplot(predictions[y_pred == 0, 1], line='s', fit=True)
            plt.title(f'QQ plot de probabilidad predicha para Clase 0 en {path}')


            plt.show()

            # Clase 1
            plt.figure(figsize=(6, 6))
            qqplot(predictions[y_pred == 1, 1], line='s', fit=True)
            plt.title(f'QQ plot de probabilidad predicha para Clase 1 en {path}')
            plt.show()




            fpr, tpr, _ = roc_curve(y_test[:, 1], predictions[:, 1])
            roc_auc = auc(fpr, tpr)
            roc_data.append((fpr, tpr, roc_auc, path))

        except Exception as e:
            print(f"Error al evaluar el modelo {path}: {str(e)}")

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