from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('api/stats', views.get_stats, name='get_stats'),
    path('api/upload', views.upload_source, name='upload_source'),
]
