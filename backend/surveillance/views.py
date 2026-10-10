import cv2
import threading
import time
import os
import numpy as np
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO

# Global state for MVP demo
current_person_count = 0
current_status = "MONITORING"
lock = threading.Lock()
camera = None
current_source = 0  # 0 means webcam, string means file path, None means off
source_name = "CAMERA 01 (Webcam)"

print("Loading YOLO11n model... (this may take a moment)")
model = YOLO('yolo11n.pt')

def get_frame():
    global current_person_count, current_status, camera, current_source
    
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
        if local_source is not None:
            frame = cv2.resize(frame, (640, 480))
            results = model(frame, classes=[0], verbose=False)
            person_count = 0
            for r in results:
                boxes = r.boxes
                person_count += len(boxes)
                frame = r.plot()
            
        with lock:
            current_person_count = person_count
            if local_source is None:
                current_status = "OFFLINE"
            elif person_count > 0:
                current_status = "PERSON DETECTED"
            else:
                current_status = "MONITORING"
                
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
            'status': current_status,
            'source_name': source_name
        })

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
