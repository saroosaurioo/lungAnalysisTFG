import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import csv

#Cargar y preprocesar las imagenes
def load_and_preprocess_data(data_path, nueva_etiqueta, target_size=(224, 224)):
    imagenes = []
    labels = []
    for category, label in nueva_etiqueta.items():
        category_path = os.path.join(data_path, category)
        for img in os.listdir(category_path):
            img_path = os.path.join(category_path, img)
            image = load_img(img_path, target_size=target_size)
            image = img_to_array(image)
            image = preprocess_input(image)
            imagenes.append(image)
            labels.append(label)

    imagenes = np.array(imagenes)
    labels = to_categorical(np.array(labels), num_classes=2)  # Asegúrate de que el número de clases sea correcto
    return imagenes, labels


def evaluate_and_plot(model, X_test, y_test, model_name):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"{model_name} - Accuracy: {accuracy * 100:.2f}%")

    # Generar predicciones y matriz de confusión
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Patologico'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, y_test[:, 1], y_pred[:, 1], model_name


# Configuraciones
data_path = r'..\..\..\dataset'
nueva_etiqueta = {'No_findings': 0, 'Covid-19': 1, 'Pneumonia': 1}

# Cargar y preparar datos
X, y = load_and_preprocess_data(data_path, nueva_etiqueta)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Cargar modelos preentrenados
model1 = load_model('modelo2_vgg3_base.h5')
model2 = load_model('modelo2_vgg3_dataAugmentation.h5')

# Evaluar y visualizar resultados
results = []

print("Evaluación Modelo 1:")
fpr1, tpr1, roc_auc1, y_test1, y_pred1, model_name1 = evaluate_and_plot(model1, X_test, y_test, 'Modelo 1: VGG Base')
results.append({'Modelo': model_name1, 'y_test': y_test1, 'predictions': y_pred1})

print("Evaluación Modelo 2:")
fpr2, tpr2, roc_auc2, y_test2, y_pred2, model_name2 = evaluate_and_plot(model2, X_test, y_test,
                                                                        'Modelo 2: VGG con Data Augmentation')
results.append({'Modelo': model_name2, 'y_test': y_test2, 'predictions': y_pred2})

# Guardar los resultados en un archivo CSV
with open('resultados.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='|')
    writer.writerow(['Modelo', 'y_test', 'predictions'])
    for result in results:
        for yt, pred in zip(result['y_test'], result['predictions']):
            writer.writerow([result['Modelo'], yt, pred])

# Graficar las curvas ROC de ambos modelos en la misma gráfica
plt.figure()
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'Modelo 1 ROC curve (area = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, color='blue', lw=2, label=f'Modelo 2 ROC curve (area = {roc_auc2:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Comparación de Modelos')
plt.legend(loc='lower right')
plt.show()
