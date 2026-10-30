import cv2
import threading
import time
import os
import json
import numpy as np
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO
from .models import SecurityIncident

ZONE_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'zone_config.json')

def load_zone_config():
    default_config = {
        'mode': 'right',
        'custom_box': {'x1': 320, 'y1': 0, 'x2': 640, 'y2': 480}
    }
    if os.path.exists(ZONE_CONFIG_FILE):
        try:
            with open(ZONE_CONFIG_FILE, 'r') as f:
                data = json.load(f)
                return {
                    'mode': data.get('mode', 'right'),
                    'custom_box': data.get('custom_box', default_config['custom_box'])
                }
        except Exception as e:
            print("Error loading zone_config.json:", e)
    return default_config

def save_zone_config(mode, custom_box):
    try:
        with open(ZONE_CONFIG_FILE, 'w') as f:
            json.dump({'mode': mode, 'custom_box': custom_box}, f, indent=2)
    except Exception as e:
        print("Error saving zone_config.json:", e)

# Initial load from disk
saved_cfg = load_zone_config()

# Global state for surveillance and threat detection
current_person_count = 0
current_intruder_count = 0
current_weapon_count = 0
current_status = "MONITORING"
current_zone_mode = saved_cfg['mode']
current_custom_box = saved_cfg['custom_box']
last_snapshot_time = 0
lock = threading.Lock()
camera = None
current_source = 0  # 0 means webcam, string means file path, None means off
source_name = "CAMERA 01 (Webcam)"

# Restricted Zone Presets on standard 640x480 frame (covers full height 0-480)
ZONE_PRESETS = {
    'right': np.array([[320, 0], [640, 0], [640, 480], [320, 480]], np.int32),
    'left': np.array([[0, 0], [320, 0], [320, 480], [0, 480]], np.int32),
    'center': np.array([[160, 0], [480, 0], [480, 480], [160, 480]], np.int32),
    'corridor': np.array([[180, 0], [460, 0], [580, 480], [60, 480]], np.int32),
    'off': None
}

def is_person_in_zone(x1, y1, x2, y2, zone_polygon):
    """
    Precision Head/Upper-Body & Majority Overlap Detection:
    - Primary Anchor: Head & Face (upper 22% center of mass).
      A person is physically in a zone when their head/face enters it.
    - Secondary Anchor: Torso / Chest center.
    - Volumetric Overlap: Requires >= 45% of total body volume inside,
      ensuring background items (curtains, pillows, chairs) never trigger false alarms.
    """
    if zone_polygon is None:
        return False, (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.22))
        
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    
    # 1. Head / Face Anchor (top 22% of bounding box, centered horizontally)
    head_x = int((x1 + x2) / 2)
    head_y = int(y1 + h * 0.22)
    
    if cv2.pointPolygonTest(zone_polygon, (head_x, head_y), False) >= 0:
        return True, (head_x, head_y)
        
    # 2. Torso Center of Mass Anchor (middle of body)
    torso_x = head_x
    torso_y = int(y1 + h * 0.50)
    
    if cv2.pointPolygonTest(zone_polygon, (torso_x, torso_y), False) >= 0:
        return True, (head_x, head_y)
        
    # 3. Ground / Base Stance Anchor (standing feet or seated base)
    base_y = min(max(y2 - 15, 0), 479)
    if cv2.pointPolygonTest(zone_polygon, (head_x, base_y), False) >= 0:
        return True, (head_x, head_y)

    # 4. Majority Overlap Check (Requires >= 45% of total body volume inside)
    gx_start = x1 + int(w * 0.05)
    gx_end = x2 - int(w * 0.05)
    gy_start = y1 + int(h * 0.05)
    gy_end = y2 - int(h * 0.05)
    
    inside_pts = 0
    total_pts = 25
    x_steps = [int(gx_start + i * (gx_end - gx_start) / 4) for i in range(5)]
    y_steps = [int(gy_start + j * (gy_end - gy_start) / 4) for j in range(5)]
    
    for px in x_steps:
        for py in y_steps:
            if cv2.pointPolygonTest(zone_polygon, (px, py), False) >= 0:
                inside_pts += 1
                
    overlap_ratio = inside_pts / float(total_pts)
    if overlap_ratio >= 0.45:
        return True, (head_x, head_y)
        
    return False, (head_x, head_y)

print("Loading YOLO11n person model... (this may take a moment)")
model = YOLO('yolo11n.pt')

print("Loading Weapon detection model (Pistol & Knife)...")
weapon_model = YOLO('weapon_yolov8n.pt')

def get_frame():
    global current_person_count, current_intruder_count, current_weapon_count, current_status
    global camera, current_source, current_zone_mode, last_snapshot_time
    
    local_source = current_source
    is_image = isinstance(local_source, str) and local_source.lower().endswith(('.png', '.jpg', '.jpeg'))
    is_video_file = isinstance(local_source, str) and not is_image
    static_frame = None
    
    # State for frame skipping & speed optimization on uploaded videos
    frame_idx = 0
    cached_detected_persons = []
    cached_detected_weapons = []
    cached_person_count = 0
    cached_intruder_count = 0
    cached_weapon_count = 0
    
    # State tracking: previous counts to detect actual ENTRY/WEAPON events
    prev_intruder_count = 0
    prev_weapon_count = 0
    
    if is_image:
        static_frame = cv2.imread(local_source)
    elif local_source is not None:
        if camera is not None:
            camera.release()
        camera = cv2.VideoCapture(local_source)
        
    while True:
        # Check if source has changed globally
        if local_source != current_source:
            local_source = current_source
            is_image = isinstance(local_source, str) and local_source.lower().endswith(('.png', '.jpg', '.jpeg'))
            is_video_file = isinstance(local_source, str) and not is_image
            frame_idx = 0
            cached_detected_persons = []
            cached_detected_weapons = []
            cached_person_count = 0
            cached_intruder_count = 0
            cached_weapon_count = 0
            prev_intruder_count = 0
            prev_weapon_count = 0
            if is_image:
                if camera is not None:
                    camera.release()
                    camera = None
                static_frame = cv2.imread(local_source)
            elif local_source is not None:
                if camera is not None:
                    camera.release()
                camera = cv2.VideoCapture(local_source)
            else:
                if camera is not None:
                    camera.release()
                    camera = None
                
        # Get frame depending on source type
        if local_source is None:
            # Camera turned off
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA OFFLINE", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            person_count = 0
            intruder_count = 0
            time.sleep(0.5)
        elif is_image:
            if static_frame is None:
                time.sleep(1)
                continue
            frame = static_frame.copy()
            time.sleep(0.1) # Simulate FPS for image
        else:
            if camera is None or not camera.isOpened():
                time.sleep(1)
                continue
                
            success, frame = camera.read()
            if not success:
                # Video ended, loop it
                if isinstance(local_source, str):
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = camera.read()
                    if not success:
                        time.sleep(0.1)
                        continue
                else:
                    time.sleep(0.1)
                    continue

        # Resize and Infer (only if camera is ON)
        person_count = 0
        intruder_count = 0
        weapon_count = 0
        detected_weapons = []
        
        if local_source is not None:
            frame = cv2.resize(frame, (640, 480))
            frame_idx += 1
            
            with lock:
                active_zone_key = current_zone_mode
                custom_box = current_custom_box.copy()
            
            if active_zone_key == 'custom':
                bx1 = custom_box['x1']
                by1 = custom_box['y1']
                bx2 = custom_box['x2']
                by2 = custom_box['y2']
                zone_polygon = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], np.int32)
            else:
                zone_polygon = ZONE_PRESETS.get(active_zone_key)
            
            # For CPU efficiency: run person detection on frame % 2 == 1, weapon detection on frame % 2 == 0
            # For video files: run every 3rd frame
            run_person_ai = (frame_idx % 2 == 1) if not is_video_file else (frame_idx % 3 == 1)
            run_weapon_ai = (frame_idx % 2 == 0) if not is_video_file else (frame_idx % 3 == 2)
            
            # --- 1. Person Detection ---
            if run_person_ai or not cached_detected_persons:
                results = model(frame, classes=[0], imgsz=480, verbose=False)
                
                detected_persons = []
                for r in results:
                    for box in r.boxes:
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        is_in_zone, anchor_pt = is_person_in_zone(x1, y1, x2, y2, zone_polygon)
                        
                        if is_in_zone:
                            intruder_count += 1
                            
                        detected_persons.append({
                            'coords': (x1, y1, x2, y2),
                            'conf': conf,
                            'anchor': anchor_pt,
                            'is_intruder': is_in_zone
                        })
                cached_detected_persons = detected_persons
                cached_person_count = person_count
                cached_intruder_count = intruder_count
            else:
                detected_persons = []
                intruder_count = 0
                person_count = len(cached_detected_persons)
                for p in cached_detected_persons:
                    x1, y1, x2, y2 = p['coords']
                    is_in_zone, anchor_pt = is_person_in_zone(x1, y1, x2, y2, zone_polygon)
                    if is_in_zone:
                        intruder_count += 1
                    detected_persons.append({
                        'coords': (x1, y1, x2, y2),
                        'conf': p['conf'],
                        'anchor': anchor_pt,
                        'is_intruder': is_in_zone
                    })
                cached_intruder_count = intruder_count

            # --- 2. Weapon Detection (Pistol & Knife) ---
            if run_weapon_ai or (frame_idx <= 2):
                w_results = weapon_model(frame, conf=0.40, imgsz=480, verbose=False)
                detected_weapons = []
                for r in w_results:
                    for box in r.boxes:
                        wx1, wy1, wx2, wy2 = map(int, box.xyxy[0])
                        w_conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = weapon_model.names.get(cls_id, 'weapon').upper()
                        weapon_count += 1
                        detected_weapons.append({
                            'coords': (wx1, wy1, wx2, wy2),
                            'conf': w_conf,
                            'name': cls_name
                        })
                cached_detected_weapons = detected_weapons
                cached_weapon_count = weapon_count
            else:
                detected_weapons = cached_detected_weapons
                weapon_count = cached_weapon_count
            
            # 1. Render Restricted Zone overlay
            if zone_polygon is not None:
                overlay = frame.copy()
                zone_color = (0, 0, 180) if intruder_count > 0 else (0, 90, 200) # Crimson when breached, amber when safe
                cv2.fillPoly(overlay, [zone_polygon], zone_color)
                alpha = 0.35 if intruder_count > 0 else 0.20
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                # Perimeter border
                border_color = (0, 0, 255) if intruder_count > 0 else (0, 165, 255)
                cv2.polylines(frame, [zone_polygon], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)
                
                # Zone label
                pts = zone_polygon
                min_x = max(10, int(np.min(pts[:, 0])))
                min_y = max(30, int(np.min(pts[:, 1])) + 28)
                label = "RESTRICTED ZONE [BREACH!]" if intruder_count > 0 else "RESTRICTED ZONE"
                cv2.rectangle(frame, (min_x, min_y - 24), (min_x + 230, min_y), border_color, -1)
                cv2.putText(frame, label, (min_x + 6, min_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 2. Render Person Bounding Boxes
            for p in detected_persons:
                x1, y1, x2, y2 = p['coords']
                conf = p['conf']
                hx, hy = p['anchor']
                is_intruder = p['is_intruder']
                
                if is_intruder:
                    box_color = (0, 0, 255)  # Bright Red for Intruder
                    tag = f"INTRUDER {conf*100:.0f}%"
                    cv2.circle(frame, (hx, hy), 6, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (hx, hy), 12, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    box_color = (0, 220, 100)  # Vibrant Green for Normal Person
                    tag = f"Person {conf*100:.0f}%"
                    cv2.circle(frame, (hx, hy), 4, (0, 220, 100), -1, cv2.LINE_AA)
                    
                # Clean bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
                
                # Corner bracket accents for modern surveillance look
                corner_len = min(16, int((x2 - x1) * 0.15))
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 3, cv2.LINE_AA)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 3, cv2.LINE_AA)
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, 3, cv2.LINE_AA)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, 3, cv2.LINE_AA)
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, 3, cv2.LINE_AA)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, 3, cv2.LINE_AA)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 3, cv2.LINE_AA)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 3, cv2.LINE_AA)
                
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, max(15, y1 - 20)), (x1 + tw + 8, max(15, y1)), box_color, -1)
                cv2.putText(frame, tag, (x1 + 4, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                
            # 3. Render Weapon Bounding Boxes (Neon Orange / Red alert style)
            for w in detected_weapons:
                wx1, wy1, wx2, wy2 = w['coords']
                w_tag = f"WEAPON: {w['name']} {w['conf']*100:.0f}%"
                w_color = (0, 69, 255) # Bright Neon Orange-Red
                cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), w_color, 2, cv2.LINE_AA)
                (wtw, wth), _ = cv2.getTextSize(w_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (wx1, max(15, wy1 - 20)), (wx1 + wtw + 8, max(15, wy1)), w_color, -1)
                cv2.putText(frame, w_tag, (wx1 + 4, max(12, wy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # 4. Flashing Alert Banner and Evidence Capture
            is_weapon_threat = weapon_count > 0
            is_intrusion_threat = intruder_count > 0
            
            if is_weapon_threat:
                cv2.rectangle(frame, (0, 0), (640, 30), (0, 35, 220), -1)
                cv2.putText(frame, f"CRITICAL: {weapon_count} WEAPON DETECTED ({detected_weapons[0]['name']})", (15, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Auto-save snapshot on new weapon appearance
                is_new_weapon = (prev_weapon_count == 0 and weapon_count > 0) or (weapon_count > prev_weapon_count)
                if is_new_weapon and (time.time() - last_snapshot_time > 2.0):
                    last_snapshot_time = time.time()
                    try:
                        evidence_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'media', 'evidence')
                        os.makedirs(evidence_dir, exist_ok=True)
                        snap_name = f"weapon_{int(time.time())}.jpg"
                        snap_path = os.path.join(evidence_dir, snap_name)
                        cv2.imwrite(snap_path, frame)

                        SecurityIncident.objects.create(
                            threat_type=f"WEAPON DETECTED ({detected_weapons[0]['name']})",
                            severity='CRITICAL',
                            source=source_name,
                            intruder_count=weapon_count,
                            snapshot=f"evidence/{snap_name}",
                            details=f"Weapon detected with {detected_weapons[0]['conf']*100:.1f}% confidence."
                        )
                    except Exception as err:
                        print("Weapon incident record error:", err)

            elif is_intrusion_threat:
                cv2.rectangle(frame, (0, 0), (640, 30), (0, 0, 220), -1)
                cv2.putText(frame, f"CRITICAL: {intruder_count} INTRUSION EVENT(S) IN RESTRICTED AREA", (15, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
                
                is_new_entry = (prev_intruder_count == 0 and intruder_count > 0) or (intruder_count > prev_intruder_count)
                if is_new_entry and (time.time() - last_snapshot_time > 2.0):
                    last_snapshot_time = time.time()
                    try:
                        evidence_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'media', 'evidence')
                        os.makedirs(evidence_dir, exist_ok=True)
                        snap_name = f"intrusion_{int(time.time())}.jpg"
                        snap_path = os.path.join(evidence_dir, snap_name)
                        cv2.imwrite(snap_path, frame)

                        SecurityIncident.objects.create(
                            threat_type='UNAUTHORIZED INTRUSION',
                            severity='CRITICAL',
                            source=source_name,
                            intruder_count=intruder_count,
                            snapshot=f"evidence/{snap_name}",
                            details=f"New entry detected: {intruder_count} intruder(s) entered restricted sector ({current_zone_mode.upper()})."
                        )
                    except Exception as err:
                        print("Evidence/Incident record error:", err)
            
            prev_intruder_count = intruder_count
            prev_weapon_count = weapon_count
            
        with lock:
            current_person_count = person_count
            current_intruder_count = intruder_count
            current_weapon_count = weapon_count
            if local_source is None:
                current_status = "OFFLINE"
            elif weapon_count > 0:
                current_status = f"WEAPON DETECTED ({detected_weapons[0]['name'] if detected_weapons else 'ARMED'})"
            elif intruder_count > 0:
                current_status = "INTRUSION DETECTED"
            elif person_count > 0:
                current_status = "MONITORING (PERSON DETECTED)"
            else:
                current_status = "MONITORING (CLEAR)"
                
        # Encode and yield
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
    return render(request, 'surveillance/dashboard.html')

@auth_required
def upload_dashboard(request):
    return render(request, 'surveillance/upload_dashboard.html')

@auth_required
def logs_dashboard(request):
    incidents = SecurityIncident.objects.all().order_by('-timestamp')
    total_incidents = incidents.count()
    return render(request, 'surveillance/logs_dashboard.html', {
        'incidents': incidents,
        'total_incidents': total_incidents
    })

@auth_required
def video_feed(request):
    return StreamingHttpResponse(get_frame(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_stats(request):
    with lock:
        return JsonResponse({
            'person_count': current_person_count,
            'intruder_count': current_intruder_count,
            'weapon_count': current_weapon_count,
            'alert_active': (current_intruder_count > 0 or current_weapon_count > 0),
            'threat_type': 'WEAPON' if current_weapon_count > 0 else ('INTRUSION' if current_intruder_count > 0 else 'NONE'),
            'zone_mode': current_zone_mode,
            'custom_box': current_custom_box,
            'status': current_status,
            'source_name': source_name
        })

def set_zone(request):
    global current_zone_mode
    if request.method == 'POST':
        zone = request.POST.get('zone', 'right')
        if zone in ZONE_PRESETS or zone == 'custom':
            with lock:
                current_zone_mode = zone
                custom_box = current_custom_box.copy()
            save_zone_config(zone, custom_box)
            return JsonResponse({'status': 'success', 'zone': zone})
    return JsonResponse({'status': 'error', 'message': 'Invalid zone preset'})

def set_custom_zone(request):
    global current_zone_mode, current_custom_box
    if request.method == 'POST':
        try:
            x1 = int(float(request.POST.get('x1', 320)))
            y1 = int(float(request.POST.get('y1', 0)))
            x2 = int(float(request.POST.get('x2', 640)))
            y2 = int(float(request.POST.get('y2', 480)))
            
            x_min = max(0, min(640, min(x1, x2)))
            x_max = max(0, min(640, max(x1, x2)))
            y_min = max(0, min(480, min(y1, y2)))
            y_max = max(0, min(480, max(y1, y2)))
            
            if x_max - x_min < 20:
                x_max = min(640, x_min + 50)
            if y_max - y_min < 20:
                y_max = min(480, y_min + 50)
                
            new_box = {'x1': x_min, 'y1': y_min, 'x2': x_max, 'y2': y_max}
            with lock:
                current_custom_box = new_box
                current_zone_mode = 'custom'
            save_zone_config('custom', new_box)
            return JsonResponse({'status': 'success', 'custom_box': new_box, 'zone': 'custom'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

def toggle_camera(request):
    global current_source, source_name
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'on':
            current_source = 0
            source_name = "CAMERA 01 (Webcam)"
        else:
            current_source = None
            source_name = "OFFLINE"
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'})

def upload_source(request):
    global current_source, source_name
    if request.method == 'POST':
        if 'file' in request.FILES:
            file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)
            
            current_source = file_path
            source_name = f"FILE: {filename}"
            return JsonResponse({'status': 'success', 'message': f'Switched to {filename}'})
            
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

@auth_required
def delete_incident(request):
    if request.method == 'POST':
        incident_id = request.POST.get('incident_id')
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
                    except Exception as e:
                        pass
            incidents.delete()
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
@auth_required
def get_analytics(request):
    from django.utils import timezone
    from datetime import timedelta
    from django.db.models import Count

    now = timezone.now()
    seven_days_ago = now - timedelta(days=6)

    # 1. Threat Type Breakdown
    all_incidents = SecurityIncident.objects.all()
    total_count = all_incidents.count()
    
    weapon_count = all_incidents.filter(threat_type__icontains='WEAPON').count()
    intrusion_count = all_incidents.filter(threat_type__icontains='INTRUSION').count()
    other_count = max(0, total_count - weapon_count - intrusion_count)

    # 2. Last 7 Days Incident Trend
    daily_labels = []
    daily_values = []
    for i in range(7):
        day_date = (seven_days_ago + timedelta(days=i)).date()
        daily_labels.append(day_date.strftime('%b %d'))
        day_count = all_incidents.filter(timestamp__date=day_date).count()
        daily_values.append(day_count)

    # 3. Peak Threat Hours (Grouped by 4-hour windows)
    hourly_labels = ["00-04h", "04-08h", "08-12h", "12-16h", "16-20h", "20-24h"]
    hourly_values = [0] * 6
    for inc in all_incidents:
        h = inc.timestamp.hour
        window_idx = min(5, h // 4)
        hourly_values[window_idx] += 1

    return JsonResponse({
        'total': total_count,
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
        }
    })

