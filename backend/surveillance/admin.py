from django.contrib import admin
from .models import Camera, SecurityIncident

@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'camera_type', 'stream_url', 'location', 'status', 'is_active', 'last_seen')
    list_filter = ('camera_type', 'status', 'is_active')
    search_fields = ('name', 'location', 'stream_url')

@admin.register(SecurityIncident)
class SecurityIncidentAdmin(admin.ModelAdmin):
    list_display = ('id', 'threat_type', 'severity', 'source', 'camera', 'intruder_count', 'timestamp')
    list_filter = ('severity', 'threat_type', 'camera')
    search_fields = ('threat_type', 'source', 'details')
