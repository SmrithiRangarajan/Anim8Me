from django.urls import path
from . import views
from .views_api import LandmarkDataAPI 

urlpatterns = [
    path('', views.home, name='home'),
    path('webcam/', views.webcam_view, name='webcam'),
    path('webcam_with_pose/', views.webcam_with_pose, name='webcam_with_pose'),
    path('pose_estimation/', views.pose_estimation, name='pose_estimation'),
    path('api/landmarks/', LandmarkDataAPI.as_view(), name='landmarks-api'),
    path('motion_visualization/', views.motion_visualization, name='motion-visualization'),
    path('replay_motion/', views.replay_motion, name='replay-motion')
    
]
