# mainapp/serializers.py

from rest_framework import serializers

class LandmarkDataSerializer(serializers.Serializer):
    landmarks = serializers.ListField()
