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
    path('api/set-zone', views.set_zone, name='set_zone'),
    path('api/set-custom-zone', views.set_custom_zone, name='set_custom_zone'),
    path('api/delete-incident', views.delete_incident, name='delete_incident'),
    path('api/clear-all-incidents', views.clear_all_incidents, name='clear_all_incidents'),
    path('api/analytics', views.get_analytics, name='get_analytics'),
]
