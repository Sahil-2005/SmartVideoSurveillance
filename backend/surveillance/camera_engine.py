"""
High-Performance Multi-Camera Surveillance Engine.
Features:
- Separate camera capture, visual rendering, and shared AI inference.
- One dedicated shared YOLO InferenceWorker with fair round-robin scheduling across all active cameras.
- CPU optimization for Intel i3-1005G1 (limiting PyTorch threads, downscaled inference, cached box interpolation).
- Zero latency build-up buffer management for RTSP / IP Webcam phone streams.
- Resilient auto-reconnection for mobile devices.
- State-change evidence snapshots (prevents duplicate disk fills).
"""

import cv2
import threading
import time
import os
import json
import numpy as np
from datetime import datetime
# Configure OpenCV ffmpeg capture timeout (5s in microseconds)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000000")
# Optimize PyTorch CPU threading before importing YOLO
try:
    import torch
    torch.set_num_threads(2)  # Leave remaining CPU threads for capture & Django
except Exception as e:
    print("PyTorch thread configuration note:", e)

from ultralytics import YOLO

# Standard Zone Presets on 640x480 frame
ZONE_PRESETS = {
    'right': np.array([[320, 0], [640, 0], [640, 480], [320, 480]], np.int32),
    'left': np.array([[0, 0], [320, 0], [320, 480], [0, 480]], np.int32),
    'center': np.array([[160, 0], [480, 0], [480, 480], [160, 480]], np.int32),
    'corridor': np.array([[180, 0], [460, 0], [580, 480], [60, 480]], np.int32),
    'off': None
}

def get_zone_polygon(zone_mode, zone_config=None):
    """Resolve zone mode and custom box into a numpy polygon array."""
    if zone_mode == 'custom' and zone_config:
        bx1 = int(zone_config.get('x1', 320))
        by1 = int(zone_config.get('y1', 0))
        bx2 = int(zone_config.get('x2', 640))
        by2 = int(zone_config.get('y2', 480))
        return np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], np.int32)
    return ZONE_PRESETS.get(zone_mode, None)

def is_person_in_zone(x1, y1, x2, y2, zone_polygon):
    """
    Precision Head/Upper-Body & Majority Overlap Detection:
    - Primary Anchor: Head & Face (upper 22% center of mass).
    - Secondary Anchor: Torso / Chest center.
    - Volumetric Overlap: Requires >= 45% of total body volume inside.
    """
    if zone_polygon is None:
        return False, (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.22))
        
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    
    # 1. Head / Face Anchor
    head_x = int((x1 + x2) / 2)
    head_y = int(y1 + h * 0.22)
    if cv2.pointPolygonTest(zone_polygon, (head_x, head_y), False) >= 0:
        return True, (head_x, head_y)
        
    # 2. Torso Center of Mass Anchor
    torso_x = head_x
    torso_y = int(y1 + h * 0.50)
    if cv2.pointPolygonTest(zone_polygon, (torso_x, torso_y), False) >= 0:
        return True, (head_x, head_y)
        
    # 3. Ground / Base Stance Anchor
    base_y = min(max(y2 - 15, 0), 479)
    if cv2.pointPolygonTest(zone_polygon, (head_x, base_y), False) >= 0:
        return True, (head_x, head_y)

    # 4. Majority Overlap Check (>= 45% of body volume)
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


class SharedInferenceWorker:
    """
    Dedicated worker thread executing shared YOLO inference with fair
    round-robin scheduling across all active cameras.
    Ensures exactly 1 inference runs at any instant, preventing CPU spikes.
    """
    _instance = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.person_model = None
        self.weapon_model = None
        self.registered_processors = []  # list of CameraProcessor references
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self._round_robin_idx = 0
        self._inference_counter = 0

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True

            # Load models once in this worker thread
            print("[InferenceWorker] Initializing YOLO11n person & weapon models...")
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            person_pt = os.path.join(base_dir, 'yolo11n.pt')
            weapon_pt = os.path.join(base_dir, 'weapon_yolov8n.pt')

            try:
                self.person_model = YOLO(person_pt if os.path.exists(person_pt) else 'yolo11n.pt')
            except Exception as e:
                print(f"[InferenceWorker] Error loading person model: {e}")
                self.person_model = None

            try:
                self.weapon_model = YOLO(weapon_pt if os.path.exists(weapon_pt) else 'weapon_yolov8n.pt')
            except Exception as e:
                print(f"[InferenceWorker] Error loading weapon model: {e}")
                self.weapon_model = None

            self.thread = threading.Thread(target=self._run_loop, name="InferenceWorkerThread", daemon=True)
            self.thread.start()
            print("[InferenceWorker] Round-robin AI worker thread started.")

    def stop(self):
        with self.lock:
            self.running = False

    def register(self, processor):
        with self.lock:
            if processor not in self.registered_processors:
                self.registered_processors.append(processor)

    def unregister(self, processor):
        with self.lock:
            if processor in self.registered_processors:
                self.registered_processors.remove(processor)

    def _run_loop(self):
        while self.running:
            with self.lock:
                processors = list(self.registered_processors)

            if not processors:
                time.sleep(0.04)
                continue

            # Fair round-robin selection
            with self.lock:
                self._round_robin_idx = (self._round_robin_idx + 1) % len(processors)
                target_proc = processors[self._round_robin_idx]

            # Ask camera for a pending inference frame
            frame_data = target_proc.get_frame_for_inference()
            if frame_data is None:
                # Target camera has no fresh frame or is offline; small sleep to yield CPU
                time.sleep(0.01)
                continue

            frame, frame_idx = frame_data
            self._inference_counter += 1

            detected_persons = []
            detected_weapons = []

            # 1. Person Detection (classes=[0] for person)
            if self.person_model and target_proc.person_detection_enabled:
                try:
                    p_results = self.person_model(frame, classes=[0], imgsz=480, verbose=False)
                    for r in p_results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            detected_persons.append({
                                'coords': (x1, y1, x2, y2),
                                'conf': conf
                            })
                except Exception as e:
                    print(f"[InferenceWorker] Person detection error on Cam {target_proc.camera_id}: {e}")

            # 2. Weapon Detection (Knife & Pistol)
            # Efficient CPU scheduling: run weapon detection if enabled and
            # either a person is detected or on alternate cycles
            should_run_weapon = (
                self.weapon_model and 
                target_proc.weapon_detection_enabled and 
                (len(detected_persons) > 0 or (self._inference_counter % 2 == 0))
            )
            if should_run_weapon:
                try:
                    w_results = self.weapon_model(frame, conf=0.40, imgsz=480, verbose=False)
                    for r in w_results:
                        for box in r.boxes:
                            wx1, wy1, wx2, wy2 = map(int, box.xyxy[0])
                            w_conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            cls_name = self.weapon_model.names.get(cls_id, 'weapon').upper()
                            detected_weapons.append({
                                'coords': (wx1, wy1, wx2, wy2),
                                'conf': w_conf,
                                'name': cls_name
                            })
                except Exception as e:
                    print(f"[InferenceWorker] Weapon detection error on Cam {target_proc.camera_id}: {e}")

            # Feed detection results back to camera processor
            target_proc.receive_inference_results(detected_persons, detected_weapons, frame_idx)


class CameraProcessor:
    """
    Independent capture and visual rendering pipeline for a single camera.
    Decoupled from AI inference and Django web request threads.
    """

    def __init__(self, camera_id, name, stream_url, camera_type='ip_webcam', location='',
                 zone_mode='right', zone_config=None, person_detection=True, weapon_detection=True):
        self.camera_id = camera_id
        self.name = name
        self.stream_url = str(stream_url).strip()
        self.camera_type = camera_type
        self.location = location

        self.zone_mode = zone_mode
        self.zone_config = zone_config or {'x1': 320, 'y1': 0, 'x2': 640, 'y2': 480}
        self.person_detection_enabled = person_detection
        self.weapon_detection_enabled = weapon_detection

        # Thread synchronization
        self.lock = threading.Lock()
        self.running = False
        self.capture_thread = None
        self.render_thread = None

        # Capture & frame state
        self.latest_raw_frame = None
        self.latest_jpeg = None
        self.raw_frame_id = 0
        self.last_inferred_frame_id = -1
        self.status = "CONNECTING"
        self.last_seen = None

        # Detection cache & counts
        self.cached_persons = []
        self.cached_weapons = []
        self.person_count = 0
        self.intruder_count = 0
        self.weapon_count = 0

        # State change tracking to prevent duplicate snapshots
        self.prev_intruder_count = 0
        self.prev_weapon_count = 0
        self.last_snapshot_time = 0

        # Create offline standby frame
        self._standby_jpeg = self._create_standby_frame("CONNECTING...")

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True

        # Register with shared AI worker
        SharedInferenceWorker.get_instance().register(self)

        # Start capture loop and rendering loop
        self.capture_thread = threading.Thread(target=self._capture_loop, name=f"CamCapture-{self.camera_id}", daemon=True)
        self.capture_thread.start()

        self.render_thread = threading.Thread(target=self._render_loop, name=f"CamRender-{self.camera_id}", daemon=True)
        self.render_thread.start()
        print(f"[CameraProcessor {self.camera_id}] Pipeline started for '{self.name}' ({self.stream_url}).")

    def stop(self):
        with self.lock:
            self.running = False
        SharedInferenceWorker.get_instance().unregister(self)
        print(f"[CameraProcessor {self.camera_id}] Stopped.")

    def update_settings(self, name=None, stream_url=None, location=None, zone_mode=None,
                        zone_config=None, person_detection=None, weapon_detection=None):
        with self.lock:
            if name is not None:
                self.name = name
            if stream_url is not None and str(stream_url).strip() != self.stream_url:
                self.stream_url = str(stream_url).strip()
                # Need to restart capture with new stream URL
                self._need_reconnect = True
            if location is not None:
                self.location = location
            if zone_mode is not None:
                self.zone_mode = zone_mode
            if zone_config is not None:
                self.zone_config = zone_config
            if person_detection is not None:
                self.person_detection_enabled = person_detection
            if weapon_detection is not None:
                self.weapon_detection_enabled = weapon_detection

    def _resolve_capture_source(self):
        """Parse source url into int (for webcam) or string (for URL/file)."""
        url = self.stream_url.strip()
        if url.isdigit():
            return int(url)
        return url

    def _create_standby_frame(self, message="NO SIGNAL"):
        """Generate high-tech dark standby frame for offline/reconnecting states."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Background gradient / grid lines
        for y in range(0, 480, 40):
            cv2.line(frame, (0, y), (640, y), (20, 24, 33), 1)
        for x in range(0, 640, 40):
            cv2.line(frame, (x, 0), (x, 480), (20, 24, 33), 1)

        # Outer border
        cv2.rectangle(frame, (20, 20), (620, 460), (45, 55, 72), 1)

        # Camera Header
        cam_tag = f"{self.name.upper()} [ID: {self.camera_id}]"
        cv2.putText(frame, cam_tag, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1, cv2.LINE_AA)
        
        # Status icon / text
        cv2.circle(frame, (320, 200), 38, (30, 41, 59), -1)
        cv2.circle(frame, (320, 200), 38, (239, 68, 68) if "OFFLINE" in message else (245, 158, 11), 2)
        cv2.putText(frame, "!", (313, 212), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, message, (320 - len(message)*8, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (241, 245, 249), 2, cv2.LINE_AA)
        
        hint = f"Source: {self.stream_url}"
        cv2.putText(frame, hint[:50], (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 116, 139), 1, cv2.LINE_AA)
        
        t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, t_str, (470, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1, cv2.LINE_AA)

        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ret else None

    def _capture_loop(self):
        """
        Background capture loop.
        Critically drops stale frames for network streams (IP Webcam, RTSP)
        so there is zero latency accumulation!
        """
        self._need_reconnect = False

        while self.running:
            source = self._resolve_capture_source()
            is_image = isinstance(source, str) and source.lower().endswith(('.png', '.jpg', '.jpeg'))
            is_file = isinstance(source, str) and not source.startswith(('http://', 'https://', 'rtsp://')) and not is_image

            if is_image:
                img = cv2.imread(source)
                if img is not None:
                    with self.lock:
                        self.latest_raw_frame = img
                        self.raw_frame_id += 1
                        self.status = "ONLINE"
                        self.last_seen = datetime.now()
                    while self.running and not self._need_reconnect:
                        time.sleep(0.1)
                else:
                    with self.lock:
                        self.status = "OFFLINE"
                        self.latest_jpeg = self._create_standby_frame("IMAGE NOT FOUND")
                    time.sleep(3)
                continue

            # Open video stream / webcam
            with self.lock:
                self.status = "CONNECTING"
            
            if isinstance(source, int):
                import platform
                if platform.system() == 'Windows':
                    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(source)
                else:
                    cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source)

            # Performance optimizations for network streams
            if isinstance(source, str) and source.startswith(('http://', 'https://', 'rtsp://')):
                # Prefer small buffer for low latency
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

            if not cap.isOpened():
                with self.lock:
                    self.status = "RECONNECTING"
                    self.latest_jpeg = self._create_standby_frame("RECONNECTING STREAM...")
                cap.release()
                time.sleep(3.0)  # Reconnection cooldown
                continue

            with self.lock:
                self.status = "ONLINE"
                self.last_seen = datetime.now()
                self._need_reconnect = False

            consecutive_failures = 0

            while self.running and not self._need_reconnect:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if is_file:
                        # Video file ended: loop back to beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.04)
                        continue
                    if consecutive_failures > 5:
                        # Stream lost
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                with self.lock:
                    self.latest_raw_frame = frame
                    self.raw_frame_id += 1
                    self.last_seen = datetime.now()
                    self.status = "ONLINE"

                # If reading from a local file, simulate normal playback rate (~25 fps)
                if is_file:
                    time.sleep(0.035)

            cap.release()
            with self.lock:
                self.status = "RECONNECTING"
                self.latest_jpeg = self._create_standby_frame("RECONNECTING...")
            time.sleep(2.0)

    def get_frame_for_inference(self):
        """Called by SharedInferenceWorker to get latest pending frame for AI."""
        with self.lock:
            if not self.running or self.status != "ONLINE" or self.latest_raw_frame is None:
                return None
            if self.raw_frame_id == self.last_inferred_frame_id:
                # Already ran inference on this frame
                return None
            frame_copy = cv2.resize(self.latest_raw_frame, (640, 480))
            frame_id = self.raw_frame_id
        return frame_copy, frame_id

    def receive_inference_results(self, detected_persons, detected_weapons, frame_idx):
        """Called by SharedInferenceWorker with fresh detection results."""
        with self.lock:
            self.cached_persons = detected_persons
            self.cached_weapons = detected_weapons
            self.last_inferred_frame_id = frame_idx

    def _render_loop(self):
        """
        Visual rendering loop running at ~20-25 FPS.
        Draws bounding boxes, zone overlay, alerts, performs state-change snapshot logging,
        and encodes JPEG for web clients.
        """
        from surveillance.models import SecurityIncident, Camera

        while self.running:
            with self.lock:
                raw_frame = self.latest_raw_frame
                status = self.status
                zone_mode = self.zone_mode
                zone_config = self.zone_config.copy() if self.zone_config else {}
                cached_p = list(self.cached_persons)
                cached_w = list(self.cached_weapons)

            if raw_frame is None or status != "ONLINE":
                time.sleep(0.05)
                continue

            display_frame = cv2.resize(raw_frame, (640, 480))
            zone_polygon = get_zone_polygon(zone_mode, zone_config)

            person_count = len(cached_p)
            intruder_count = 0
            detected_persons_with_zone = []

            for p in cached_p:
                x1, y1, x2, y2 = p['coords']
                conf = p['conf']
                is_in_zone, anchor_pt = is_person_in_zone(x1, y1, x2, y2, zone_polygon)
                if is_in_zone:
                    intruder_count += 1
                detected_persons_with_zone.append({
                    'coords': (x1, y1, x2, y2),
                    'conf': conf,
                    'anchor': anchor_pt,
                    'is_intruder': is_in_zone
                })

            weapon_count = len(cached_w)

            # 1. Render Restricted Zone overlay
            if zone_polygon is not None:
                overlay = display_frame.copy()
                zone_color = (0, 0, 180) if intruder_count > 0 else (0, 90, 200)
                cv2.fillPoly(overlay, [zone_polygon], zone_color)
                alpha = 0.35 if intruder_count > 0 else 0.20
                cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

                # Perimeter border
                border_color = (0, 0, 255) if intruder_count > 0 else (0, 165, 255)
                cv2.polylines(display_frame, [zone_polygon], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

                # Zone label badge
                pts = zone_polygon
                min_x = max(10, int(np.min(pts[:, 0])))
                min_y = max(30, int(np.min(pts[:, 1])) + 28)
                label = f"ZONE: {zone_mode.upper()} [BREACH!]" if intruder_count > 0 else f"ZONE: {zone_mode.upper()}"
                cv2.rectangle(display_frame, (min_x, min_y - 24), (min_x + 210, min_y), border_color, -1)
                cv2.putText(display_frame, label, (min_x + 6, min_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # 2. Render Person Bounding Boxes
            for p in detected_persons_with_zone:
                x1, y1, x2, y2 = p['coords']
                conf = p['conf']
                hx, hy = p['anchor']
                is_intruder = p['is_intruder']

                if is_intruder:
                    box_color = (0, 0, 255)  # Bright Red
                    tag = f"INTRUDER {conf*100:.0f}%"
                    cv2.circle(display_frame, (hx, hy), 6, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(display_frame, (hx, hy), 12, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    box_color = (0, 220, 100)  # Vibrant Green
                    tag = f"Person {conf*100:.0f}%"
                    cv2.circle(display_frame, (hx, hy), 4, (0, 220, 100), -1, cv2.LINE_AA)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

                # Modern corner accents
                c_len = min(14, int((x2 - x1) * 0.15))
                cv2.line(display_frame, (x1, y1), (x1 + c_len, y1), box_color, 3, cv2.LINE_AA)
                cv2.line(display_frame, (x1, y1), (x1, y1 + c_len), box_color, 3, cv2.LINE_AA)
                cv2.line(display_frame, (x2, y1), (x2 - c_len, y1), box_color, 3, cv2.LINE_AA)
                cv2.line(display_frame, (x2, y1), (x2, y1 + c_len), box_color, 3, cv2.LINE_AA)
                cv2.line(display_frame, (x1, y2), (x1 + c_len, y2), box_color, 3, cv2.LINE_AA)
                cv2.line(display_frame, (x1, y2), (x1, y2 - c_len), box_color, 3, cv2.LINE_AA)
                cv2.line(display_frame, (x2, y2), (x2 - c_len, y2), box_color, 3, cv2.LINE_AA)
                cv2.line(display_frame, (x2, y2), (x2, y2 - c_len), box_color, 3, cv2.LINE_AA)

                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(display_frame, (x1, max(15, y1 - 20)), (x1 + tw + 8, max(15, y1)), box_color, -1)
                cv2.putText(display_frame, tag, (x1 + 4, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Render Weapon Bounding Boxes
            for w in cached_w:
                wx1, wy1, wx2, wy2 = w['coords']
                w_tag = f"WEAPON: {w['name']} {w['conf']*100:.0f}%"
                w_color = (0, 69, 255)  # Neon Orange-Red
                cv2.rectangle(display_frame, (wx1, wy1), (wx2, wy2), w_color, 2, cv2.LINE_AA)
                (wtw, wth), _ = cv2.getTextSize(w_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(display_frame, (wx1, max(15, wy1 - 20)), (wx1 + wtw + 8, max(15, wy1)), w_color, -1)
                cv2.putText(display_frame, w_tag, (wx1 + 4, max(12, wy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # 4. Flashing Alert Banners & Smart Snapshot Recording
            is_weapon_threat = weapon_count > 0
            is_intrusion_threat = intruder_count > 0

            if is_weapon_threat:
                cv2.rectangle(display_frame, (0, 0), (640, 28), (0, 35, 220), -1)
                w_name = cached_w[0]['name'] if cached_w else 'WEAPON'
                cv2.putText(display_frame, f"CRITICAL: {weapon_count} WEAPON DETECTED ({w_name}) - {self.name}",
                            (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

                # Only save on state change (new weapon appearance) with 2.0s cooldown
                is_new_weapon = (self.prev_weapon_count == 0 and weapon_count > 0) or (weapon_count > self.prev_weapon_count)
                if is_new_weapon and (time.time() - self.last_snapshot_time > 2.0):
                    self.last_snapshot_time = time.time()
                    try:
                        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        evidence_dir = os.path.join(base_dir, 'media', 'evidence')
                        os.makedirs(evidence_dir, exist_ok=True)
                        snap_name = f"cam{self.camera_id}_weapon_{int(time.time())}.jpg"
                        snap_path = os.path.join(evidence_dir, snap_name)
                        cv2.imwrite(snap_path, display_frame)

                        cam_obj = Camera.objects.filter(id=self.camera_id).first()
                        SecurityIncident.objects.create(
                            camera=cam_obj,
                            threat_type=f"WEAPON DETECTED ({w_name})",
                            severity='CRITICAL',
                            source=f"{self.name} ({self.location or 'Camera feed'})",
                            intruder_count=weapon_count,
                            snapshot=f"evidence/{snap_name}",
                            details=f"Weapon detected on camera #{self.camera_id} with {cached_w[0]['conf']*100:.1f}% confidence."
                        )
                    except Exception as err:
                        print(f"[CameraProcessor {self.camera_id}] Error logging weapon incident:", err)

            elif is_intrusion_threat:
                cv2.rectangle(display_frame, (0, 0), (640, 28), (0, 0, 220), -1)
                cv2.putText(display_frame, f"CRITICAL: {intruder_count} INTRUSION EVENT(S) - {self.name}",
                            (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

                # State change snapshot
                is_new_entry = (self.prev_intruder_count == 0 and intruder_count > 0) or (intruder_count > self.prev_intruder_count)
                if is_new_entry and (time.time() - self.last_snapshot_time > 2.0):
                    self.last_snapshot_time = time.time()
                    try:
                        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        evidence_dir = os.path.join(base_dir, 'media', 'evidence')
                        os.makedirs(evidence_dir, exist_ok=True)
                        snap_name = f"cam{self.camera_id}_intrusion_{int(time.time())}.jpg"
                        snap_path = os.path.join(evidence_dir, snap_name)
                        cv2.imwrite(snap_path, display_frame)

                        cam_obj = Camera.objects.filter(id=self.camera_id).first()
                        SecurityIncident.objects.create(
                            camera=cam_obj,
                            threat_type='UNAUTHORIZED INTRUSION',
                            severity='CRITICAL',
                            source=f"{self.name} ({self.location or 'Restricted Sector'})",
                            intruder_count=intruder_count,
                            snapshot=f"evidence/{snap_name}",
                            details=f"Intruder breach detected on camera #{self.camera_id} in {zone_mode.upper()} sector."
                        )
                    except Exception as err:
                        print(f"[CameraProcessor {self.camera_id}] Error logging intrusion incident:", err)

            self.prev_intruder_count = intruder_count
            self.prev_weapon_count = weapon_count

            # Small HUD footer in camera view
            cv2.rectangle(display_frame, (0, 456), (640, 480), (15, 23, 42), -1)
            hud_text = f"CAM {self.camera_id}: {self.name}  |  PEOPLE: {person_count}  |  INTRUDERS: {intruder_count}  |  WEAPONS: {weapon_count}"
            cv2.putText(display_frame, hud_text, (10, 473), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)

            # Update public state
            with self.lock:
                self.person_count = person_count
                self.intruder_count = intruder_count
                self.weapon_count = weapon_count
                if weapon_count > 0:
                    self.status = f"WEAPON ALERT ({cached_w[0]['name']})"
                elif intruder_count > 0:
                    self.status = "INTRUSION ALERT"
                elif person_count > 0:
                    self.status = "MONITORING (PERSON)"
                else:
                    self.status = "MONITORING (CLEAR)"

            # Encode JPEG for clients
            ret, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                with self.lock:
                    self.latest_jpeg = buf.tobytes()

            time.sleep(0.04)  # ~25 FPS visual rendering

    def generate_mjpeg(self):
        """Generator that yields MJPEG frames for StreamingHttpResponse."""
        while True:
            with self.lock:
                if self.running:
                    frame_bytes = self.latest_jpeg or self._standby_jpeg
                else:
                    frame_bytes = self._standby_jpeg
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.04)

    def get_stats(self):
        """Thread-safe snapshot of camera detection stats."""
        with self.lock:
            return {
                'camera_id': self.camera_id,
                'name': self.name,
                'stream_url': self.stream_url,
                'camera_type': self.camera_type,
                'location': self.location,
                'status': self.status,
                'person_count': self.person_count,
                'intruder_count': self.intruder_count,
                'weapon_count': self.weapon_count,
                'alert_active': (self.intruder_count > 0 or self.weapon_count > 0),
                'zone_mode': self.zone_mode,
                'zone_config': self.zone_config,
                'last_seen': self.last_seen.strftime("%H:%M:%S") if self.last_seen else None
            }


class CameraManager:
    """
    Central Manager managing all active CameraProcessor instances.
    Coordinates lifecycle, starts/stops processors, aggregates stats,
    and handles testing camera connectivity.
    """
    _instance = None
    _init_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.processors = {}  # {camera_id: CameraProcessor}
        self.lock = threading.Lock()
        self._synced = False

    def initialize(self):
        """Start shared inference worker and synchronize active cameras from DB."""
        SharedInferenceWorker.get_instance().start()
        self.sync_from_db()

    def sync_from_db(self):
        """Load or create default camera and start processors for all active cameras."""
        from surveillance.models import Camera

        with self.lock:
            # Ensure at least 1 default camera exists for backwards compatibility
            if not Camera.objects.exists():
                Camera.objects.create(
                    name="Laptop / Primary Webcam",
                    stream_url="0",
                    camera_type="webcam",
                    location="Control Room",
                    is_active=True,
                    zone_mode="right",
                    zone_config={'x1': 320, 'y1': 0, 'x2': 640, 'y2': 480}
                )

            active_cameras = Camera.objects.filter(is_active=True)
            active_ids = set()

            for cam in active_cameras:
                active_ids.add(cam.id)
                if cam.id not in self.processors:
                    proc = CameraProcessor(
                        camera_id=cam.id,
                        name=cam.name,
                        stream_url=cam.stream_url,
                        camera_type=cam.camera_type,
                        location=cam.location,
                        zone_mode=cam.zone_mode,
                        zone_config=cam.zone_config,
                        person_detection=cam.person_detection,
                        weapon_detection=cam.weapon_detection
                    )
                    proc.start()
                    self.processors[cam.id] = proc
                else:
                    # Update settings
                    self.processors[cam.id].update_settings(
                        name=cam.name,
                        stream_url=cam.stream_url,
                        location=cam.location,
                        zone_mode=cam.zone_mode,
                        zone_config=cam.zone_config,
                        person_detection=cam.person_detection,
                        weapon_detection=cam.weapon_detection
                    )

            # Stop processors for deactivated or deleted cameras
            to_remove = [cid for cid in self.processors if cid not in active_ids]
            for cid in to_remove:
                proc = self.processors.pop(cid)
                proc.stop()

            self._synced = True

    def get_processor(self, camera_id):
        """Retrieve processor specifically for camera_id (never falls back to a different camera)."""
        with self.lock:
            if not self._synced:
                self.sync_from_db()

            if camera_id in self.processors:
                return self.processors[camera_id]

            # Try loading from DB if it exists
            from surveillance.models import Camera
            cam = Camera.objects.filter(id=camera_id).first()
            if cam:
                proc = CameraProcessor(
                    camera_id=cam.id,
                    name=cam.name,
                    stream_url=cam.stream_url,
                    camera_type=cam.camera_type,
                    location=cam.location,
                    zone_mode=cam.zone_mode,
                    zone_config=cam.zone_config,
                    person_detection=cam.person_detection,
                    weapon_detection=cam.weapon_detection
                )
                if cam.is_active:
                    proc.start()
                else:
                    proc.status = "DISABLED"
                    proc._standby_jpeg = proc._create_standby_frame("CAMERA DISABLED")
                self.processors[cam.id] = proc
                return proc

            return None

    def start_camera(self, camera_obj):
        """Start or restart a camera processor from a Camera model instance."""
        with self.lock:
            if camera_obj.id in self.processors:
                self.processors[camera_obj.id].stop()

            proc = CameraProcessor(
                camera_id=camera_obj.id,
                name=camera_obj.name,
                stream_url=camera_obj.stream_url,
                camera_type=camera_obj.camera_type,
                location=camera_obj.location,
                zone_mode=camera_obj.zone_mode,
                zone_config=camera_obj.zone_config,
                person_detection=camera_obj.person_detection,
                weapon_detection=camera_obj.weapon_detection
            )
            proc.start()
            self.processors[camera_obj.id] = proc
            return proc

    def stop_camera(self, camera_id):
        """Stop and remove a camera processor."""
        with self.lock:
            proc = self.processors.pop(camera_id, None)
            if proc:
                proc.stop()

    def get_all_stats(self):
        """Aggregate detection statistics across all active cameras."""
        with self.lock:
            if not self._synced:
                self.sync_from_db()
            procs = list(self.processors.values())

        total_persons = 0
        total_intruders = 0
        total_weapons = 0
        online_count = 0
        camera_stats = []

        for p in procs:
            stats = p.get_stats()
            camera_stats.append(stats)
            if "ONLINE" in stats['status'] or "ALERT" in stats['status'] or "MONITORING" in stats['status']:
                online_count += 1
            total_persons += stats['person_count']
            total_intruders += stats['intruder_count']
            total_weapons += stats['weapon_count']

        if total_weapons > 0:
            system_status = f"CRITICAL: {total_weapons} WEAPON(S) DETECTED"
        elif total_intruders > 0:
            system_status = f"WARNING: {total_intruders} ZONE INTRUSION(S)"
        elif total_persons > 0:
            system_status = f"ACTIVE: {total_persons} PERSON(S) MONITORED"
        elif online_count > 0:
            system_status = "SYSTEM NORMAL - ALL CLEAR"
        else:
            system_status = "ALL CAMERAS OFFLINE"

        return {
            'total_cameras': len(procs),
            'online_cameras': online_count,
            'total_persons': total_persons,
            'total_intruders': total_intruders,
            'total_weapons': total_weapons,
            'alert_active': (total_intruders > 0 or total_weapons > 0),
            'system_status': system_status,
            'cameras': camera_stats
        }

    @staticmethod
    def test_connection(stream_url, camera_type='ip_webcam'):
        """
        Quick non-blocking probe to test if a camera URL or device is reachable.
        Returns dictionary with success status, resolution, and error details.
        """
        url_str = str(stream_url).strip()
        source = int(url_str) if url_str.isdigit() else url_str

        try:
            start_t = time.time()
            if isinstance(source, int):
                import platform
                if platform.system() == 'Windows':
                    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(source)
                else:
                    cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return {
                    'success': False,
                    'error': 'Could not open stream. Check IP address, port, and Wi-Fi connection.'
                }

            ret, frame = cap.read()
            cap.release()
            latency_ms = int((time.time() - start_t) * 1000)

            if not ret or frame is None:
                return {
                    'success': False,
                    'error': 'Connected to stream but failed to read initial video frame.'
                }

            h, w = frame.shape[:2]
            return {
                'success': True,
                'width': w,
                'height': h,
                'latency_ms': latency_ms,
                'message': f"Connected successfully! Stream active ({w}x{h} @ ~{latency_ms}ms ping)."
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def get_camera_manager():
    """Access the global CameraManager singleton."""
    return CameraManager.get_instance()
