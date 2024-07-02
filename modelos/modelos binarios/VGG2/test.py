import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import csv


# Función para cargar y preparar datos
def load_and_preprocess_data(data_path, labels_dict, target_size=(224, 224)):
    images = []
    labels = []
    for label, indices in labels_dict.items():
        label_path = os.path.join(data_path, label)
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            image = load_img(img_path, target_size=target_size)
            image = img_to_array(image)
            image = preprocess_input(image)
            images.append(image)
            labels.append(indices)

    images = np.array(images)
    labels = to_categorical(labels, num_classes=len(set(labels_dict.values())))  # Ajusta el número de clases
    return images, labels


# Cargar datos
data_path = r'..\..\..\dataset'
labels_dict = {'No_findings': 0, 'Covid-19': 1, 'Pneumonia': 1}
images, labels = load_and_preprocess_data(data_path, labels_dict)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=42, stratify=labels)

# Cargar modelos
model1 = load_model('modelo2_vgg2_base.h5')
model2 = load_model('modelo2_vgg2_dataAugmentation.h5')


# Función para evaluar y mostrar resultados
def evaluate_and_plot(model, X, y, title):
    loss, accuracy = model.evaluate(X, y, verbose=0)
    print(f'{title} - Accuracy: {accuracy * 100:.2f}%')
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y, axis=1)
    cm = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Patologico'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusión - {title}')
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(true_classes, predictions[:, 1])
    roc_auc = auc(fpr, tpr)

    # Devolver los valores necesarios para el archivo CSV
    return fpr, tpr, roc_auc, true_classes, predictions[:, 1], title


# Evaluar modelos y guardar resultados
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
