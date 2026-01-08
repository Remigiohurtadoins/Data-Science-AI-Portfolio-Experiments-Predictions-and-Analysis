from datetime import datetime
import base64, secrets, io
from PIL import Image
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from apiCNN.Logica import modeloCNN
from apiCNN.models import Imagen
from django.core import serializers


def obtener_imagen(data_url, base_dir):
    _format, _dataurl = data_url.split(';base64,')
    _filename, _extension = secrets.token_hex(20), _format.split('/')[-1]
    if _extension == 'octet-stream':
        _extension = 'png'
    file = ContentFile(base64.b64decode(_dataurl), name=f"{_filename}.{_extension}")
    image = Image.open(file)
    image_io = io.BytesIO()
    image = image.resize((150, 150), Image.ANTIALIAS)
    image.save(image_io, format=_extension)
    file = ContentFile(image_io.getvalue(), name=f"{_filename}.{_extension}")
    fss = FileSystemStorage()
    dtm = datetime.today()
    time = str(int(dtm.timestamp()))
    _filename += time + '.' + _extension
    fss.save(_filename, file)
    prediccion = modeloCNN.predecir_imagen(base_dir + '/media/' + _filename)
    obj_imagen = Imagen(int(dtm.timestamp()), _filename, prediccion.get('pred'),
                        round(float(prediccion.get('prob')) * 100, 2))
    obj_imagen.save()
    return prediccion


def listar_registros():
    imagenes = Imagen.objects.all()
    imagenes_json = serializers.serialize('json', imagenes)
    print(imagenes_json)
    return imagenes_json
