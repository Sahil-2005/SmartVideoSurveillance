from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('', views.dashboard, name='dashboard'),
    path('upload-dashboard', views.upload_dashboard, name='upload_dashboard'),
    path('logs-dashboard', views.logs_dashboard, name='logs_dashboard'),
    path('analytics-dashboard', views.analytics_dashboard, name='analytics_dashboard'),
    
    # Video Streaming Feeds
    path('video_feed', views.video_feed, name='video_feed'),
    path('video_feed/<int:camera_id>/', views.video_feed_camera, name='video_feed_camera'),
    
    # Global & Legacy APIs
    path('api/stats', views.get_stats, name='get_stats'),
    path('api/upload', views.upload_source, name='upload_source'),
    path('api/clear-upload', views.clear_upload_source, name='clear_upload_source'),
    path('api/toggle-camera', views.toggle_camera, name='toggle_camera'),
    path('api/set-zone', views.set_zone, name='set_zone'),
    path('api/set-custom-zone', views.set_custom_zone, name='set_custom_zone'),
    path('api/delete-incident', views.delete_incident, name='delete_incident'),
    path('api/clear-all-incidents', views.clear_all_incidents, name='clear_all_incidents'),
    path('api/analytics', views.get_analytics, name='get_analytics'),

    # Multi-Camera Management APIs
    path('api/cameras', views.camera_list, name='camera_list'),
    path('api/cameras/add', views.camera_add, name='camera_add'),
    path('api/cameras/<int:camera_id>/edit', views.camera_edit, name='camera_edit'),
    path('api/cameras/<int:camera_id>/delete', views.camera_delete, name='camera_delete'),
    path('api/cameras/<int:camera_id>/toggle', views.camera_toggle, name='camera_toggle'),
    path('api/cameras/<int:camera_id>/stats', views.camera_stats, name='camera_stats'),
    path('api/cameras/<int:camera_id>/set-zone', views.camera_set_zone, name='camera_set_zone'),
    path('api/cameras/test', views.camera_test, name='camera_test'),
]

