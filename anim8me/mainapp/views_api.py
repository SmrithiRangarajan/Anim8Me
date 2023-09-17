# mainapp/views_api.py

from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import LandmarkDataSerializer
from .views import landmark_data  # Import the landmark_data variable

class LandmarkDataAPI(APIView):
    def get(self, request):
        serializer = LandmarkDataSerializer({'landmarks': landmark_data})
        return Response(serializer.data)
