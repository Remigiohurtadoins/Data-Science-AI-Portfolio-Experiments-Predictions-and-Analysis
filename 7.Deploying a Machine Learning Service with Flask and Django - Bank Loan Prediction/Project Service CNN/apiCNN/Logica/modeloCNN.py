import numpy as np
from keras.preprocessing.image import load_img
from tensorflow.keras.models import model_from_json
from keras.preprocessing import image
from keras import backend as k
from tensorflow.keras.utils import plot_model


def predecir_imagen(img_url):
    print(img_url)
    tam_img = (128, 128)
    url_modelo= r'apiCNN/Logica/modelo'
    url_pesos = r'apiCNN/Logica/pesos'
    modelo = cargar_rnn(url_modelo, url_pesos)
    img = load_img(img_url, target_size=tam_img)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255
    resultado = modelo.predict(img)
    probabilidad_gato = resultado.tolist()[0][0]
    probabilidad_perro = resultado.tolist()[0][1]
    datos = dict()
    if probabilidad_gato > probabilidad_perro:
        datos['pred'] = 'GATO'
        datos['prob'] = probabilidad_gato
        return datos
    else:
        datos['pred'] = 'PERRO'
        datos['prob'] = probabilidad_perro
        return datos


def cargar_rnn(nombreArchivoModelo, nombreArchivoPesos):
    k.reset_uids()
    # Cargar la Arquitectura desde el archivo JSON
    with open(nombreArchivoModelo + '.json', 'r') as f:
        model = model_from_json(f.read())
    # Cargar Pesos (weights) en el nuevo modelo
    model.load_weights(nombreArchivoPesos + '.h5')
    return model
