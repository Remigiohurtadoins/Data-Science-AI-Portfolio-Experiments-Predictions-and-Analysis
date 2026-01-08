from rest_framework import serializers
from apiCNN import models


class ImagenSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            'id',
            'name',
            'animal',
            'percentage'
        )
        model = models.Imagen

