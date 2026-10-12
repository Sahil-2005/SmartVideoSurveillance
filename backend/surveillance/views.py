import cv2
import threading
import time
import os
import numpy as np
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO

# Global state for surveillance and threat detection
current_person_count = 0
current_intruder_count = 0
current_status = "MONITORING"
current_zone_mode = 'right'  # 'right', 'left', 'center', 'corridor', 'off'
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
    Precision Intrusion Detection:
    Eliminates false alarms caused by loose bounding box margins, arm gestures, or chairs.
    Only triggers when the person's core center of mass (torso/chest) or a substantial
    portion (>35%) of their inner body core has entered the restricted zone.
    """
    if zone_polygon is None:
        return False
        
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    
    # 1. Primary Check: Core Center of Mass (Torso / Chest)
    if cv2.pointPolygonTest(zone_polygon, (cx, cy), False) >= 0:
        return True
        
    # 2. Check Lower Torso / Stance Center (seated base or standing feet)
    base_y = min(max(y2 - 20, 0), 479)
    if cv2.pointPolygonTest(zone_polygon, (cx, base_y), False) >= 0:
        return True

    # 3. Inner Core Multi-Point Check:
    # Sample points strictly within the inner body core (inner 50% width and height)
    w = x2 - x1
    h = y2 - y1
    inner_x1 = x1 + int(w * 0.28)
    inner_x2 = x2 - int(w * 0.28)
    inner_y1 = y1 + int(h * 0.20)
    inner_y2 = y2 - int(h * 0.20)
    
    core_samples = [
        (cx, cy),
        (inner_x1, cy),
        (inner_x2, cy),
        (cx, inner_y1),
        (cx, inner_y2),
        (inner_x1, inner_y2),
        (inner_x2, inner_y2)
    ]
    
    inside_count = sum(1 for pt in core_samples if cv2.pointPolygonTest(zone_polygon, pt, False) >= 0)
    # Require at least 3 core body points inside to confirm a true intrusion
    return inside_count >= 3

print("Loading YOLO11n model... (this may take a moment)")
model = YOLO('yolo11n.pt')

def get_frame():
    global current_person_count, current_intruder_count, current_status
    global camera, current_source, current_zone_mode, last_snapshot_time
    
    local_source = current_source
    is_image = isinstance(local_source, str) and local_source.lower().endswith(('.png', '.jpg', '.jpeg'))
    static_frame = None
    
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
        if local_source is not None:
            frame = cv2.resize(frame, (640, 480))
            
            with lock:
                active_zone_key = current_zone_mode
            zone_polygon = ZONE_PRESETS.get(active_zone_key)
            
            # YOLO11n inference for Person (class 0)
            results = model(frame, classes=[0], verbose=False)
            
            detected_persons = []
            for r in results:
                for box in r.boxes:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Tracked body center anchor
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    is_in_zone = is_person_in_zone(x1, y1, x2, y2, zone_polygon)
                    
                    if is_in_zone:
                        intruder_count += 1
                        
                    detected_persons.append({
                        'coords': (x1, y1, x2, y2),
                        'conf': conf,
                        'center': (cx, cy),
                        'is_intruder': is_in_zone
                    })
            
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
                cx, cy = p['center']
                is_intruder = p['is_intruder']
                
                if is_intruder:
                    box_color = (0, 0, 255)  # Bright Red for Intruder
                    tag = f"INTRUDER {conf*100:.0f}%"
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 12, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    box_color = (0, 220, 100)  # Green for Normal Person
                    tag = f"Person {conf*100:.0f}%"
                    cv2.circle(frame, (cx, cy), 4, (0, 220, 100), -1, cv2.LINE_AA)
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, max(15, y1 - 20)), (x1 + tw + 8, max(15, y1)), box_color, -1)
                cv2.putText(frame, tag, (x1 + 4, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                
            # 3. Flashing Alert Banner and Evidence Capture
            if intruder_count > 0:
                cv2.rectangle(frame, (0, 0), (640, 30), (0, 0, 220), -1)
                cv2.putText(frame, f"CRITICAL: {intruder_count} INTRUSION EVENT(S) IN RESTRICTED AREA", (15, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Auto-save evidence snapshot (rate-limited to 1 every 4s)
                if time.time() - last_snapshot_time > 4.0:
                    last_snapshot_time = time.time()
                    try:
                        evidence_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'media', 'evidence')
                        os.makedirs(evidence_dir, exist_ok=True)
                        snap_name = f"intrusion_{int(time.time())}.jpg"
                        snap_path = os.path.join(evidence_dir, snap_name)
                        cv2.imwrite(snap_path, frame)
                    except Exception as err:
                        print("Evidence capture error:", err)
            
        with lock:
            current_person_count = person_count
            current_intruder_count = intruder_count
            if local_source is None:
                current_status = "OFFLINE"
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
    return render(request, 'surveillance/logs_dashboard.html')

@auth_required
def video_feed(request):
    return StreamingHttpResponse(get_frame(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_stats(request):
    with lock:
        return JsonResponse({
            'person_count': current_person_count,
            'intruder_count': current_intruder_count,
            'alert_active': current_intruder_count > 0,
            'zone_mode': current_zone_mode,
            'status': current_status,
            'source_name': source_name
        })

def set_zone(request):
    global current_zone_mode
    if request.method == 'POST':
        zone = request.POST.get('zone', 'right')
        if zone in ZONE_PRESETS:
            with lock:
                current_zone_mode = zone
            return JsonResponse({'status': 'success', 'zone': zone})
    return JsonResponse({'status': 'error', 'message': 'Invalid zone preset'})

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
