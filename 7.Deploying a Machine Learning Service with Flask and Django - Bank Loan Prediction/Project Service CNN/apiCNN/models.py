from django.db.models import Model
from django.db.models import CharField
from django.db.models import FloatField


class Imagen(Model):
    image_name = CharField(max_length=100, blank=False)
    animal_predicted = CharField(max_length=30, blank=False)
    animal_percentage = FloatField()
