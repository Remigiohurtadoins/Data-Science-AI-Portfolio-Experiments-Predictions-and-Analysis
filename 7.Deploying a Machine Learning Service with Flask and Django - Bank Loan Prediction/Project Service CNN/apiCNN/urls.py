from django.conf.urls import url
from apiCNN import views

urlpatterns = [
    url(r'^$', views.Initial.main_page),
    url(r'^predecir/', views.Clasificacion.upload_image),
    url(r'^listar/', views.Clasificacion.get_results),
    url(r'^informe/', views.Initial.informe),
]
