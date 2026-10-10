from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('', views.dashboard, name='dashboard'),
    path('upload-dashboard', views.upload_dashboard, name='upload_dashboard'),
    path('logs-dashboard', views.logs_dashboard, name='logs_dashboard'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('api/stats', views.get_stats, name='get_stats'),
    path('api/upload', views.upload_source, name='upload_source'),
    path('api/toggle-camera', views.toggle_camera, name='toggle_camera'),
]
