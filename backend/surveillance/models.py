from django.db import models

class Camera(models.Model):
    STATUS_CHOICES = [
        ('online', 'Online'),
        ('offline', 'Offline'),
        ('reconnecting', 'Reconnecting'),
        ('error', 'Error'),
    ]

    TYPE_CHOICES = [
        ('ip_webcam', 'IP Webcam (Android Phone)'),
        ('rtsp', 'RTSP Stream / CCTV'),
        ('webcam', 'Local / USB Webcam'),
        ('upload', 'Uploaded Video File'),
    ]

    name = models.CharField(max_length=100)
    stream_url = models.CharField(max_length=500, help_text="e.g. 0 for webcam or http://192.168.1.5:8080/video")
    camera_type = models.CharField(max_length=20, choices=TYPE_CHOICES, default='ip_webcam')
    location = models.CharField(max_length=200, blank=True, default='')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='offline')
    is_active = models.BooleanField(default=True)
    zone_mode = models.CharField(max_length=20, default='right')
    zone_config = models.JSONField(default=dict, blank=True)
    person_detection = models.BooleanField(default=True)
    weapon_detection = models.BooleanField(default=True)
    fire_smoke_detection = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_seen = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['id']

    def __str__(self):
        return f"{self.name} ({self.get_camera_type_display()})"


class SecurityIncident(models.Model):
    SEVERITY_CHOICES = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('CRITICAL', 'Critical'),
    ]

    camera = models.ForeignKey(Camera, on_delete=models.SET_NULL, null=True, blank=True, related_name='incidents')
    timestamp = models.DateTimeField(auto_now_add=True)
    threat_type = models.CharField(max_length=100, default='UNAUTHORIZED INTRUSION')
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='CRITICAL')
    source = models.CharField(max_length=255, default='CAMERA 01 (Webcam)')
    intruder_count = models.IntegerField(default=1)
    snapshot = models.ImageField(upload_to='evidence/', null=True, blank=True)
    details = models.TextField(blank=True, default='')

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.threat_type} at {self.timestamp:%Y-%m-%d %H:%M:%S} ({self.severity})"

