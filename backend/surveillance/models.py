from django.db import models

class SecurityIncident(models.Model):
    SEVERITY_CHOICES = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('CRITICAL', 'Critical'),
    ]

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
