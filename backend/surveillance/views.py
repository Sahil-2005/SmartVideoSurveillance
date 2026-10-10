import cv2
import threading
import time
import os
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO

# Global state for MVP demo
current_person_count = 0
current_status = "SAFE"
lock = threading.Lock()
camera = None
current_source = 0  # 0 means webcam, string means file path
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
    else:
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
            else:
                if camera is not None:
                    camera.release()
                camera = cv2.VideoCapture(local_source)
                
        # Get frame depending on source type
        if is_image:
            if static_frame is None:
                # Fallback if image failed to load
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

        # Resize for performance
        frame = cv2.resize(frame, (640, 480))
        
        # Inference
        results = model(frame, classes=[0], verbose=False)
        person_count = 0
        for r in results:
            boxes = r.boxes
            person_count += len(boxes)
            frame = r.plot()
            
        with lock:
            current_person_count = person_count
            if person_count > 0:
                current_status = "THREAT DETECTED"
            else:
                current_status = "SAFE"
                
        # Encode and yield
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def dashboard(request):
    return render(request, 'surveillance/dashboard.html')

def video_feed(request):
    return StreamingHttpResponse(get_frame(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_stats(request):
    with lock:
        return JsonResponse({
            'person_count': current_person_count,
            'status': current_status,
            'source_name': source_name
        })

def upload_source(request):
    global current_source, source_name
    if request.method == 'POST':
        source_type = request.POST.get('source_type')
        if source_type == 'webcam':
            current_source = 0
            source_name = "CAMERA 01 (Webcam)"
            return JsonResponse({'status': 'success', 'message': 'Switched to webcam'})
            
        elif 'file' in request.FILES:
            file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)
            
            current_source = file_path
            source_name = f"FILE: {filename}"
            return JsonResponse({'status': 'success', 'message': f'Switched to {filename}'})
            
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})
