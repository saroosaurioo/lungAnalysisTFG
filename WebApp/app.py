from threading import Timer

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def delete_file(filepath):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error eliminando archivo: {e}")

def analyze_image(filepath):
    modelo_clasificador = load_model('models/modelo3_vgg16.h5')
    imagen = load_img(filepath, target_size=(224, 224))  # Cargar la imagen con el tamaño adecuado
    imagen_array = img_to_array(imagen)  # Convertir la imagen a un array
    imagen_array = np.expand_dims(imagen_array, axis=0)
    imagen_array = preprocess_input(imagen_array)
    model_vgg16 = VGG16(weights='imagenet', include_top=False)
    features_imagen = model_vgg16.predict(imagen_array)
    prediccion = modelo_clasificador.predict(features_imagen)  # Predecir la clase de la imagen
    # Decodificar la prediccion (para obtener la etiqueta de la clase)
    etiqueta_predicha = np.argmax(prediccion, axis=1)
    categorias = ['Normal', 'Covid-19', 'Pneumonia']
    return categorias[etiqueta_predicha[0]]


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resultado = analyze_image(filepath)
            Timer(10, delete_file, args=[filepath]).start()

            return render_template('resultado.html', resultado=resultado,
                                   imagen_url=filepath)

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
