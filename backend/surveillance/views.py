import os
import json
import time
from datetime import datetime, timedelta
from django.shortcuts import render, redirect, get_object_or_404
from django.http import StreamingHttpResponse, JsonResponse, HttpResponseNotFound
from django.core.files.storage import FileSystemStorage
from django.db.models import Count
from django.utils import timezone

from .models import SecurityIncident, Camera
from .camera_engine import get_camera_manager, CameraManager, ZONE_PRESETS

# Ensure camera manager initialized
def _get_manager():
    mgr = get_camera_manager()
    if not mgr._synced:
        mgr.initialize()
    return mgr

def auth_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.session.get('is_authenticated'):
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return wrapper

def login_view(request):
    if request.method == 'POST':
        if request.POST.get('username') == 'admin' and request.POST.get('password') == 'admin123':
            request.session['is_authenticated'] = True
            return redirect('dashboard')
        else:
            return render(request, 'surveillance/login.html', {'error': 'Invalid credentials'})
    return render(request, 'surveillance/login.html')

def logout_view(request):
    request.session.flush()
    return redirect('login')

@auth_required
def dashboard(request):
    mgr = _get_manager()
    cameras = Camera.objects.exclude(camera_type='upload').order_by('id')
    recent_incidents = SecurityIncident.objects.all().order_by('-timestamp')[:4]
    recent_alerts = SecurityIncident.objects.all().order_by('-timestamp')[:3]
    return render(request, 'surveillance/dashboard.html', {
        'cameras': cameras,
        'camera_count': cameras.count(),
        'recent_incidents': recent_incidents,
        'recent_alerts': recent_alerts,
    })

@auth_required
def upload_dashboard(request):
    upload_cam = Camera.objects.filter(camera_type='upload', is_active=True).first()
    return render(request, 'surveillance/upload_dashboard.html', {
        'upload_cam': upload_cam
    })

@auth_required
def logs_dashboard(request):
    camera_id = request.GET.get('camera_id')
    cameras = Camera.objects.exclude(camera_type='upload').order_by('id')
    
    incidents = SecurityIncident.objects.all().order_by('-timestamp')
    if camera_id and camera_id.isdigit():
        incidents = incidents.filter(camera_id=int(camera_id))

    total_incidents = incidents.count()
    return render(request, 'surveillance/logs_dashboard.html', {
        'incidents': incidents,
        'total_incidents': total_incidents,
        'cameras': cameras,
        'selected_camera_id': int(camera_id) if camera_id and camera_id.isdigit() else None
    })

@auth_required
def analytics_dashboard(request):
    cameras = Camera.objects.exclude(camera_type='upload').order_by('id')
    camera_id = request.GET.get('camera_id')
    
    incidents = SecurityIncident.objects.all().order_by('-timestamp')
    if camera_id and camera_id.isdigit():
        incidents = incidents.filter(camera_id=int(camera_id))

    total_incidents = incidents.count()
    weapon_incidents = incidents.filter(threat_type__icontains='WEAPON').count()
    intrusion_incidents = incidents.filter(threat_type__icontains='INTRUSION').count()

    top_cam_entry = (
        incidents.filter(camera__isnull=False)
        .values('camera__name')
        .annotate(cnt=Count('id'))
        .order_by('-cnt')
        .first()
    )
    top_sector = top_cam_entry['camera__name'] if top_cam_entry else 'All Sectors Secure'

    recent_incidents = incidents[:5]

    return render(request, 'surveillance/analytics_dashboard.html', {
        'cameras': cameras,
        'selected_camera_id': int(camera_id) if camera_id and camera_id.isdigit() else None,
        'total_incidents': total_incidents,
        'weapon_incidents': weapon_incidents,
        'intrusion_incidents': intrusion_incidents,
        'top_sector': top_sector,
        'recent_incidents': recent_incidents,
    })

# --- Video Streaming Views ---

@auth_required
def video_feed(request):
    """Fallback feed streaming the primary / first active camera."""
    mgr = _get_manager()
    active_cams = Camera.objects.filter(is_active=True).exclude(camera_type='upload').order_by('id')
    primary_id = active_cams.first().id if active_cams.exists() else 1
    proc = mgr.get_processor(primary_id)
    if not proc:
        return HttpResponseNotFound("No active camera found")
    return StreamingHttpResponse(proc.generate_mjpeg(), content_type='multipart/x-mixed-replace; boundary=frame')

@auth_required
def video_feed_camera(request, camera_id):
    """Direct MJPEG stream for a specific camera ID."""
    mgr = _get_manager()
    proc = mgr.get_processor(camera_id)
    if not proc:
        return HttpResponseNotFound(f"Camera #{camera_id} not found or inactive")
    return StreamingHttpResponse(proc.generate_mjpeg(), content_type='multipart/x-mixed-replace; boundary=frame')

# --- Analytics & Telemetry ---

def get_stats(request):
    """Returns aggregated surveillance stats and per-camera breakdowns."""
    mgr = _get_manager()
    all_stats = mgr.get_all_stats(exclude_upload=True)

    # Top-level fallback properties for backwards compatibility
    primary_cam = None
    if all_stats['cameras']:
        primary_cam = all_stats['cameras'][0]

    response_data = {
        'person_count': primary_cam['person_count'] if primary_cam else 0,
        'intruder_count': primary_cam['intruder_count'] if primary_cam else 0,
        'weapon_count': primary_cam['weapon_count'] if primary_cam else 0,
        'alert_active': all_stats['alert_active'],
        'threat_type': 'FIRE' if all_stats.get('total_fire_smoke', 0) > 0 else ('WEAPON' if all_stats['total_weapons'] > 0 else ('INTRUSION' if all_stats['total_intruders'] > 0 else 'NONE')),
        'status': all_stats['system_status'],
        'total_cameras': all_stats['total_cameras'],
        'online_cameras': all_stats['online_cameras'],
        'total_persons': all_stats['total_persons'],
        'total_intruders': all_stats['total_intruders'],
        'total_weapons': all_stats['total_weapons'],
        'total_fire_smoke': all_stats.get('total_fire_smoke', 0),
        'cameras': all_stats['cameras']
    }
    return JsonResponse(response_data)

@auth_required
def get_analytics(request):
    camera_id = request.GET.get('camera_id')
    now = timezone.now()
    seven_days_ago = now - timedelta(days=6)

    all_incidents = SecurityIncident.objects.all()
    if camera_id and camera_id.isdigit():
        all_incidents = all_incidents.filter(camera_id=int(camera_id))

    total_count = all_incidents.count()
    weapon_count = all_incidents.filter(threat_type__icontains='WEAPON').count()
    intrusion_count = all_incidents.filter(threat_type__icontains='INTRUSION').count()
    other_count = max(0, total_count - weapon_count - intrusion_count)

    daily_labels = []
    daily_values = []
    for i in range(7):
        day_date = (seven_days_ago + timedelta(days=i)).date()
        daily_labels.append(day_date.strftime('%b %d'))
        day_count = all_incidents.filter(timestamp__date=day_date).count()
        daily_values.append(day_count)

    hourly_labels = ["00-04h", "04-08h", "08-12h", "12-16h", "16-20h", "20-24h"]
    hourly_values = [0] * 6
    for inc in all_incidents:
        h = inc.timestamp.hour
        window_idx = min(5, h // 4)
        hourly_values[window_idx] += 1

    cam_labels = []
    cam_values = []
    for c in Camera.objects.exclude(camera_type='upload'):
        cnt = all_incidents.filter(camera=c).count()
        cam_labels.append(c.name)
        cam_values.append(cnt)

    return JsonResponse({
        'total': total_count,
        'weapons': weapon_count,
        'intrusions': intrusion_count,
        'breakdown': {
            'labels': ['Intrusions', 'Weapons', 'Other'],
            'data': [intrusion_count, weapon_count, other_count]
        },
        'trend': {
            'labels': daily_labels,
            'data': daily_values
        },
        'hourly': {
            'labels': hourly_labels,
            'data': hourly_values
        },
        'camera_distribution': {
            'labels': cam_labels,
            'data': cam_values
        }
    })

# --- Multi-Camera Management APIs ---

@auth_required
def camera_list(request):
    """List all registered cameras with real-time stats."""
    mgr = _get_manager()
    cams = Camera.objects.all().order_by('id')
    stats_map = {c['camera_id']: c for c in mgr.get_all_stats(exclude_upload=False)['cameras']}

    result = []
    for c in cams:
        s = stats_map.get(c.id, {})
        result.append({
            'id': c.id,
            'name': c.name,
            'stream_url': c.stream_url,
            'camera_type': c.camera_type,
            'location': c.location,
            'is_active': c.is_active,
            'status': s.get('status', c.status),
            'person_count': s.get('person_count', 0),
            'intruder_count': s.get('intruder_count', 0),
            'weapon_count': s.get('weapon_count', 0),
            'fire_smoke_count': s.get('fire_smoke_count', 0),
            'person_detection': c.person_detection,
            'weapon_detection': c.weapon_detection,
            'fire_smoke_detection': c.fire_smoke_detection,
            'zone_mode': c.zone_mode,
            'zone_config': c.zone_config,
            'last_seen': s.get('last_seen', None)
        })
    return JsonResponse({'status': 'success', 'cameras': result})

@auth_required
def camera_add(request):
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        stream_url = request.POST.get('stream_url', '').strip()
        camera_type = request.POST.get('camera_type', 'ip_webcam')
        location = request.POST.get('location', '').strip()

        if not stream_url:
            return JsonResponse({'status': 'error', 'message': 'Stream URL is required.'})

        if not name:
            if location:
                name = f"Phone ({location})" if camera_type == 'ip_webcam' else f"Camera ({location})"
            else:
                name = f"Android Phone #{Camera.objects.count() + 1}" if camera_type == 'ip_webcam' else f"Security Camera #{Camera.objects.count() + 1}"

        p_detect = request.POST.get('person_detection', 'true').lower() in ('true', '1', 'on')
        w_detect = request.POST.get('weapon_detection', 'true').lower() in ('true', '1', 'on')
        fs_detect = request.POST.get('fire_smoke_detection', 'true').lower() in ('true', '1', 'on')

        cam = Camera.objects.create(
            name=name,
            stream_url=stream_url,
            camera_type=camera_type,
            location=location,
            is_active=True,
            zone_mode='right',
            zone_config={'x1': 320, 'y1': 0, 'x2': 640, 'y2': 480},
            person_detection=p_detect,
            weapon_detection=w_detect,
            fire_smoke_detection=fs_detect
        )

        mgr = _get_manager()
        mgr.start_camera(cam)

        return JsonResponse({
            'status': 'success',
            'message': f"Camera '{name}' added successfully.",
            'camera': {
                'id': cam.id,
                'name': cam.name,
                'stream_url': cam.stream_url,
                'camera_type': cam.camera_type,
                'location': cam.location
            }
        })
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})

@auth_required
def camera_edit(request, camera_id):
    if request.method == 'POST':
        cam = get_object_or_404(Camera, id=camera_id)
        name = request.POST.get('name', '').strip()
        stream_url = request.POST.get('stream_url', '').strip()
        camera_type = request.POST.get('camera_type', cam.camera_type)
        location = request.POST.get('location', cam.location).strip()

        if name:
            cam.name = name
        if stream_url:
            cam.stream_url = stream_url
        if camera_type:
            cam.camera_type = camera_type
        cam.location = location

        if 'person_detection' in request.POST:
            cam.person_detection = request.POST.get('person_detection').lower() in ('true', '1', 'on')
        if 'weapon_detection' in request.POST:
            cam.weapon_detection = request.POST.get('weapon_detection').lower() in ('true', '1', 'on')
        if 'fire_smoke_detection' in request.POST:
            cam.fire_smoke_detection = request.POST.get('fire_smoke_detection').lower() in ('true', '1', 'on')

        cam.save()

        mgr = _get_manager()
        proc = mgr.get_processor(cam.id)
        if proc:
            proc.update_settings(
                name=cam.name,
                stream_url=cam.stream_url,
                location=cam.location,
                person_detection=cam.person_detection,
                weapon_detection=cam.weapon_detection,
                fire_smoke_detection=cam.fire_smoke_detection
            )
        elif cam.is_active:
            mgr.start_camera(cam)

        return JsonResponse({'status': 'success', 'message': 'Camera updated successfully.'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})

@auth_required
def camera_delete(request, camera_id):
    if request.method == 'POST':
        cam = get_object_or_404(Camera, id=camera_id)
        mgr = _get_manager()
        mgr.stop_camera(cam.id)
        cam_name = cam.name
        cam.delete()
        return JsonResponse({'status': 'success', 'message': f"Camera '{cam_name}' removed."})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})

@auth_required
def camera_toggle(request, camera_id):
    if request.method == 'POST':
        cam = get_object_or_404(Camera, id=camera_id)
        action = request.POST.get('action') # 'on', 'off', or 'toggle'
        
        if action == 'on':
            cam.is_active = True
        elif action == 'off':
            cam.is_active = False
        else:
            cam.is_active = not cam.is_active

        cam.save()
        mgr = _get_manager()
        if cam.is_active:
            mgr.start_camera(cam)
        else:
            mgr.stop_camera(cam.id)

        return JsonResponse({'status': 'success', 'is_active': cam.is_active})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})

@auth_required
def camera_stats(request, camera_id):
    mgr = _get_manager()
    proc = mgr.get_processor(camera_id)
    if not proc:
        return JsonResponse({'status': 'error', 'message': 'Camera processor not active'})
    return JsonResponse({'status': 'success', 'stats': proc.get_stats()})

@auth_required
def camera_set_zone(request, camera_id):
    if request.method == 'POST':
        cam = get_object_or_404(Camera, id=camera_id)
        zone_mode = request.POST.get('zone', 'right')
        
        custom_box = None
        if zone_mode == 'custom':
            try:
                x1 = int(float(request.POST.get('x1', 320)))
                y1 = int(float(request.POST.get('y1', 0)))
                x2 = int(float(request.POST.get('x2', 640)))
                y2 = int(float(request.POST.get('y2', 480)))
                x_min = max(0, min(640, min(x1, x2)))
                x_max = max(0, min(640, max(x1, x2)))
                y_min = max(0, min(480, min(y1, y2)))
                y_max = max(0, min(480, max(y1, y2)))
                custom_box = {'x1': x_min, 'y1': y_min, 'x2': x_max, 'y2': y_max}
            except Exception:
                custom_box = {'x1': 320, 'y1': 0, 'x2': 640, 'y2': 480}

        cam.zone_mode = zone_mode
        if custom_box:
            cam.zone_config = custom_box
        cam.save()

        mgr = _get_manager()
        proc = mgr.get_processor(cam.id)
        if proc:
            proc.update_settings(zone_mode=zone_mode, zone_config=custom_box or cam.zone_config)

        return JsonResponse({'status': 'success', 'zone_mode': zone_mode, 'zone_config': cam.zone_config})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})

def camera_test(request):
    """Test camera URL connectivity before adding."""
    if request.method == 'POST':
        stream_url = request.POST.get('stream_url', '').strip()
        camera_type = request.POST.get('camera_type', 'ip_webcam')
        if not stream_url:
            return JsonResponse({'success': False, 'error': 'Stream URL cannot be empty.'})

        result = CameraManager.test_connection(stream_url, camera_type)
        return JsonResponse(result)
    return JsonResponse({'success': False, 'error': 'Invalid request method.'})

# --- Legacy & Compatibility Handlers ---

def set_zone(request):
    """Updates zone for primary camera (backwards compatible)."""
    if request.method == 'POST':
        camera_id = request.POST.get('camera_id')
        cam = Camera.objects.first()
        if camera_id and camera_id.isdigit():
            cam = Camera.objects.filter(id=int(camera_id)).first() or cam
        if not cam:
            return JsonResponse({'status': 'error', 'message': 'No camera found'})

        zone = request.POST.get('zone', 'right')
        cam.zone_mode = zone
        cam.save()

        mgr = _get_manager()
        proc = mgr.get_processor(cam.id)
        if proc:
            proc.update_settings(zone_mode=zone)
        return JsonResponse({'status': 'success', 'zone': zone})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

def set_custom_zone(request):
    """Updates custom zone for primary camera (backwards compatible)."""
    if request.method == 'POST':
        camera_id = request.POST.get('camera_id')
        cam = Camera.objects.first()
        if camera_id and camera_id.isdigit():
            cam = Camera.objects.filter(id=int(camera_id)).first() or cam
        if not cam:
            return JsonResponse({'status': 'error', 'message': 'No camera found'})

        try:
            x1 = int(float(request.POST.get('x1', 320)))
            y1 = int(float(request.POST.get('y1', 0)))
            x2 = int(float(request.POST.get('x2', 640)))
            y2 = int(float(request.POST.get('y2', 480)))
            box = {
                'x1': max(0, min(640, min(x1, x2))),
                'y1': max(0, min(480, min(y1, y2))),
                'x2': max(0, min(640, max(x1, x2))),
                'y2': max(0, min(480, max(y1, y2)))
            }
            cam.zone_mode = 'custom'
            cam.zone_config = box
            cam.save()

            mgr = _get_manager()
            proc = mgr.get_processor(cam.id)
            if proc:
                proc.update_settings(zone_mode='custom', zone_config=box)
            return JsonResponse({'status': 'success', 'custom_box': box, 'zone': 'custom'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

def toggle_camera(request):
    """Toggles primary camera status (backwards compatible)."""
    if request.method == 'POST':
        action = request.POST.get('action') # 'on' or 'off'
        camera_id = request.POST.get('camera_id')
        cam = Camera.objects.first()
        if camera_id and camera_id.isdigit():
            cam = Camera.objects.filter(id=int(camera_id)).first() or cam

        if cam:
            if action == 'on':
                cam.is_active = True
            elif action == 'off':
                cam.is_active = False
            else:
                cam.is_active = not cam.is_active

            cam.save()
            mgr = _get_manager()
            if cam.is_active:
                mgr.start_camera(cam)
            else:
                mgr.stop_camera(cam.id)
        return JsonResponse({'status': 'success', 'is_active': cam.is_active if cam else False})
    return JsonResponse({'status': 'error'})

def upload_source(request):
    """Uploads file and attaches to an upload camera entry."""
    if request.method == 'POST' and 'file' in request.FILES:
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        file_path = fs.path(filename)

        # Create or update upload camera
        cam, created = Camera.objects.get_or_create(
            camera_type='upload',
            defaults={
                'name': f"File: {filename}",
                'stream_url': file_path,
                'location': 'Uploaded Media',
                'is_active': True,
                'zone_mode': 'right',
                'zone_config': {'x1': 320, 'y1': 0, 'x2': 640, 'y2': 480},
                'person_detection': True,
                'weapon_detection': True,
                'fire_smoke_detection': True
            }
        )
        if not created:
            cam.name = f"File: {filename}"
            cam.stream_url = file_path
            cam.is_active = True
            cam.person_detection = True
            cam.weapon_detection = True
            cam.fire_smoke_detection = True
            cam.save()

        mgr = _get_manager()
        mgr.start_camera(cam)

        return JsonResponse({'status': 'success', 'message': f'Uploaded {filename}', 'camera_id': cam.id})
    return JsonResponse({'status': 'error', 'message': 'Invalid file upload.'})

@auth_required
def clear_upload_source(request):
    """Deactivates all uploaded media streams."""
    if request.method == 'POST':
        upload_cams = Camera.objects.filter(camera_type='upload')
        mgr = _get_manager()
        for c in upload_cams:
            c.is_active = False
            c.save()
            mgr.stop_camera(c.id)
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})

@auth_required
def delete_incident(request):
    if request.method == 'POST':
        incident_id = None
        if request.content_type and 'application/json' in request.content_type:
            try:
                import json
                data = json.loads(request.body)
                incident_id = data.get('id') or data.get('incident_id')
            except Exception:
                pass
        if not incident_id:
            incident_id = request.POST.get('incident_id') or request.POST.get('id')
            
        try:
            incident = SecurityIncident.objects.get(id=incident_id)
            if incident.snapshot:
                try:
                    if os.path.exists(incident.snapshot.path):
                        os.remove(incident.snapshot.path)
                except Exception as e:
                    print("Error removing snapshot file:", e)
            incident.delete()
            return JsonResponse({'status': 'success'})
        except SecurityIncident.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Incident not found'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

@auth_required
def clear_all_incidents(request):
    if request.method == 'POST':
        try:
            incidents = SecurityIncident.objects.all()
            for incident in incidents:
                if incident.snapshot:
                    try:
                        if os.path.exists(incident.snapshot.path):
                            os.remove(incident.snapshot.path)
                    except Exception:
                        pass
            incidents.delete()
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
