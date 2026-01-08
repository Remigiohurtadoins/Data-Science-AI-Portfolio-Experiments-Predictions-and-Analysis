from apiCNN.models import Imagen
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from apiCNN.Logica import controlador
import os
from apiCNN.Logica import modeloCNN


class Initial():
    def main_page(request):
        return render(request, "main.html")

    def informe(request):
        return render(request, 'ml_cnn.html')


class Clasificacion():
    def upload_image(request):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        BASE_DIR = BASE_DIR.replace('/apiCNN/', '/')
        if request.is_ajax() and request.method == "POST":
            data = request.POST['mydata']
            retorno = controlador.obtener_imagen(data, BASE_DIR)
            data = {
                'clase': retorno['pred'],
                'probabilidad': str(round(float(retorno['prob']), 2)*100)
            }
            return JsonResponse(data)

    def get_results(request):
        return JsonResponse(controlador.listar_registros(), safe=False)
