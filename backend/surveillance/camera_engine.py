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
    Intelligent Anatomical & Volumetric Intrusion Detection:
    - Eliminates false alarms caused by empty bounding-box corners grazing a zone.
    - True intrusion is confirmed if ANY of the following holds:
      1. Primary Anchor (Head / Face center): cv2.pointPolygonTest >= 0
      2. Secondary Anchor (Torso / Chest center): cv2.pointPolygonTest >= 0
      3. Base Anchor (Ground stance / Feet): cv2.pointPolygonTest >= 0
      4. Lateral Flanks (Mid-height torso reach at y1 + 0.50*h):
         Left flank (x1 + 0.20*w) or Right flank (x2 - 0.20*w)
      5. Significant Volumetric Overlap: >= 25% of body grid inside zone
    """
    if zone_polygon is None:
        return False, (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.25))
        
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = int((x1 + x2) / 2)
    
    # 1. Primary Head / Face Anchor (top 25% center)
    head_y = int(y1 + h * 0.25)
    if cv2.pointPolygonTest(zone_polygon, (cx, head_y), False) >= 0:
        return True, (cx, head_y)
        
    # 2. Torso Center of Mass Anchor (middle 50% center)
    torso_y = int(y1 + h * 0.50)
    if cv2.pointPolygonTest(zone_polygon, (cx, torso_y), False) >= 0:
        return True, (cx, torso_y)
        
    # 3. Ground Stance / Feet Base Anchor (bottom 5% center)
    base_y = min(max(y2 - 15, 0), 479)
    if cv2.pointPolygonTest(zone_polygon, (cx, base_y), False) >= 0:
        return True, (cx, base_y)

    # 4. Lateral Torso / Arm Breach (tested at mid-torso height, NOT empty top corners!)
    left_flank = (int(x1 + w * 0.20), torso_y)
    right_flank = (int(x2 - w * 0.20), torso_y)
    if cv2.pointPolygonTest(zone_polygon, left_flank, False) >= 0:
        return True, left_flank
    if cv2.pointPolygonTest(zone_polygon, right_flank, False) >= 0:
        return True, right_flank

    # 5. Volumetric Grid Overlap (>= 25% of body volume)
    gx_start = x1 + int(w * 0.05)
    gx_end = x2 - int(w * 0.05)
    gy_start = y1 + int(h * 0.05)
    gy_end = y2 - int(h * 0.05)
    
    inside_pts = 0
    total_pts = 25
    x_steps = [int(gx_start + i * (gx_end - gx_start) / 4) for i in range(5)]
    y_steps = [int(gy_start + j * (gy_end - gy_start) / 4) for j in range(5)]
    deepest_pt = None
    max_dist = -999999
    
    for px in x_steps:
        for py in y_steps:
            dist = cv2.pointPolygonTest(zone_polygon, (px, py), True)
            if dist >= 0:
                inside_pts += 1
                if dist > max_dist:
                    max_dist = dist
                    deepest_pt = (px, py)
                
    overlap_ratio = inside_pts / float(total_pts)
    if overlap_ratio >= 0.25:
        return True, deepest_pt or (cx, head_y)
        
    return False, (cx, head_y)


def is_complete_person(x1, y1, x2, y2, conf, frame):
    """
    Validates whether a detected bounding box represents a genuine, complete human person
    rather than background objects, a hand, an isolated arm, or visual noise.
    
    Validation Criteria:
    1. Confidence threshold: conf >= 0.40.
    2. Physical scale & area: height >= 90px, width >= 40px, area >= 5000px^2 in 640x480 frame.
    3. Aspect Ratio: height / width >= 0.60 and <= 4.5.
    4. Complete Person Stance vs Hand Discriminator:
       - Standing / walking full body: aspect >= 1.40 and height >= 90px.
       - Seated upper body (webcam): height >= 120px and aspect >= 0.65.
       - High confidence occupant: conf >= 0.65, height >= 100px, width >= 60px.
    5. Hand Rejection: Small boxes (< 18,000 px) with low-medium confidence are rejected.
    """
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
        
    # 1. Minimum confidence threshold
    if conf < 0.40:
        return False

    # 2. Minimum physical dimensions (in 640x480 resolution)
    # A complete person body/torso is at least 65px wide and 110px tall. Eliminates narrow limbs/hands
    if h < 110 or w < 65 or (w * h) < 8000:
        return False

    # 3. Valid Human Aspect Ratio
    aspect = h / float(w)
    if aspect < 0.60 or aspect > 4.0:
        return False

    # 4. Complete Person Geometry:
    is_full_body = (aspect >= 1.40 and h >= 150 and w >= 65)
    is_seated = (h >= 120 and w >= 80 and aspect >= 0.65)
    is_high_conf = (conf >= 0.65 and h >= 110 and w >= 70)
    if not (is_full_body or is_seated or is_high_conf):
        return False

    # 5. Hand Rejection:
    # A hand or isolated limb proposal is small (< 18,000 px^2) and narrow (w < 70)
    if (w * h) < 18000 and (conf < 0.65 or w < 70):
        if not is_full_body:
            return False

    return True


def filter_and_suppress_persons(raw_candidates, frame):
    """
    Applies complete person verification and Non-Maximum Suppression (NMS) /
    Containment deduplication. Eliminates duplicate boxes on the same person
    and rejects false positive objects / hands.
    """
    valid_candidates = []
    for cand in raw_candidates:
        x1, y1, x2, y2 = cand['coords']
        conf = cand['conf']
        if is_complete_person(x1, y1, x2, y2, conf, frame):
            valid_candidates.append(cand)

    if not valid_candidates:
        return []

    # Sort by confidence descending
    valid_candidates.sort(key=lambda c: c['conf'], reverse=True)

    final_persons = []
    for cand in valid_candidates:
        cx1, cy1, cx2, cy2 = cand['coords']
        cand_area = max(1, (cx2 - cx1) * (cy2 - cy1))
        is_duplicate = False

        for accepted in final_persons:
            ax1, ay1, ax2, ay2 = accepted['coords']
            acc_area = max(1, (ax2 - ax1) * (ay2 - ay1))

            # Calculate intersection
            ix1 = max(cx1, ax1)
            iy1 = max(cy1, ay1)
            ix2 = min(cx2, ax2)
            iy2 = min(cy2, ay2)

            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h

            if inter_area > 0:
                iou = inter_area / float(cand_area + acc_area - inter_area)
                containment = inter_area / float(min(cand_area, acc_area))
                # Suppress if significant IoU (> 0.30) or if one box is mostly inside another (> 0.45)
                if iou > 0.30 or containment > 0.45:
                    is_duplicate = True
                    break

        if not is_duplicate:
            final_persons.append(cand)

    return final_persons


def filter_and_suppress_weapons(raw_weapons, detected_persons=None):
    """
    Validates weapon candidates (Knife & Pistol) and applies NMS deduplication.
    
    Validation Criteria:
    1. Class-Specific Confidence Thresholds:
       - Knife: Since real knives often reflect light or have lower contrast against backgrounds,
         knives held within arm's reach of a person (<= 110px) are accepted with conf >= 0.35.
         Standalone knives away from people require conf >= 0.65.
       - Pistol: To prevent false alarms on chair corners, shadows, or shirts, pistols near a
         person require conf >= 0.52. Standalone pistols require conf >= 0.72.
    2. Physical Dimension Bounds:
       - Minimum size: width >= 16px, height >= 16px, area >= 300px^2.
       - Maximum size: area <= 75,000px^2.
    3. Aspect Ratio:
       - Rejects extreme thin lines (aspect ratio > 7.0), e.g. curtain seams, cables.
    4. Person Reach / Arm Reach Association:
       In 640x480 resolution, an outstretched arm holding a weapon extends up to 110px
       from the person's torso bounding box.
    5. Non-Maximum Suppression (IoU > 0.35): Eliminates duplicate boxes on the same weapon.
    """
    valid_candidates = []
    for w in raw_weapons:
        wx1, wy1, wx2, wy2 = w['coords']
        conf = w['conf']
        name = w.get('name', '').upper()
        ww = wx2 - wx1
        wh = wy2 - wy1
        area = ww * wh

        if ww < 16 or wh < 16 or area < 300 or area > 75000:
            continue

        aspect = max(ww, wh) / max(1, min(ww, wh))
        if aspect > 7.0:
            continue

        # Proximity to person check (arm's reach <= 110px)
        near_person = False
        if detected_persons:
            for p in detected_persons:
                px1, py1, px2, py2 = p['coords']
                # Check expanded bounding box (within arm's reach ~110px of person)
                if (wx2 >= px1 - 110 and wx1 <= px2 + 110 and wy2 >= py1 - 110 and wy1 <= py2 + 110):
                    near_person = True
                    break

        is_knife = ('KNIFE' in name)

        if near_person:
            # When wielded by a person:
            # Knives have minimal false positives on background, so threshold is sensitive (0.35)
            # Pistols require 0.52 to prevent clothing/chair fold triggers
            min_conf = 0.35 if is_knife else 0.52
            if conf < min_conf:
                continue
        else:
            # Standalone weapon away from any person:
            min_conf = 0.65 if is_knife else 0.72
            if conf < min_conf:
                continue

        valid_candidates.append(w)

    if not valid_candidates:
        return []

    # Sort descending by confidence
    valid_candidates.sort(key=lambda x: x['conf'], reverse=True)

    # NMS Deduplication
    final_weapons = []
    for cand in valid_candidates:
        cx1, cy1, cx2, cy2 = cand['coords']
        cand_area = max(1, (cx2 - cx1) * (cy2 - cy1))
        is_dup = False

        for accepted in final_weapons:
            ax1, ay1, ax2, ay2 = accepted['coords']
            acc_area = max(1, (ax2 - ax1) * (ay2 - ay1))

            ix1 = max(cx1, ax1)
            iy1 = max(cy1, ay1)
            ix2 = min(cx2, ax2)
            iy2 = min(cy2, ay2)

            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter_area > 0:
                iou = inter_area / float(cand_area + acc_area - inter_area)
                if iou > 0.35:
                    is_dup = True
                    break

        if not is_dup:
            final_weapons.append(cand)

    return final_weapons


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

            # 1. Person Detection (classes=[0] for person) with Complete Person Filtering & NMS
            if self.person_model and target_proc.person_detection_enabled:
                try:
                    p_results = self.person_model(frame, classes=[0], conf=0.40, imgsz=480, verbose=False)
                    raw_candidates = []
                    for r in p_results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            raw_candidates.append({
                                'coords': (x1, y1, x2, y2),
                                'conf': conf
                            })
                    detected_persons = filter_and_suppress_persons(raw_candidates, frame)
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
                    w_results = self.weapon_model(frame, conf=0.30, imgsz=640, verbose=False)
                    raw_weapons = []
                    for r in w_results:
                        for box in r.boxes:
                            wx1, wy1, wx2, wy2 = map(int, box.xyxy[0])
                            w_conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            cls_name = self.weapon_model.names.get(cls_id, 'weapon').upper()
                            raw_weapons.append({
                                'coords': (wx1, wy1, wx2, wy2),
                                'conf': w_conf,
                                'name': cls_name
                            })
                    detected_weapons = filter_and_suppress_weapons(raw_weapons, detected_persons)
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
        self.last_person_seen_time = 0
        self.person_count = 0
        self.intruder_count = 0
        self.weapon_count = 0

        # Weapon temporal confirmation & incident debounce
        self.weapon_threat_counter = 0
        self.last_weapon_seen_time = 0
        self.weapon_incident_active = False
        self.last_weapon_incident_time = 0

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
                # Fast pre-flight network reachability check (1.5s timeout)
                if isinstance(source, str) and source.startswith(('http://', 'https://', 'rtsp://')):
                    import urllib.parse
                    import socket
                    parsed = urllib.parse.urlparse(source)
                    if parsed.hostname:
                        port = parsed.port or (80 if parsed.scheme == 'http' else (443 if parsed.scheme == 'https' else 554))
                        try:
                            sock = socket.create_connection((parsed.hostname, port), timeout=1.5)
                            sock.close()
                        except Exception:
                            with self.lock:
                                self.status = "RECONNECTING"
                                self.latest_jpeg = self._create_standby_frame("PHONE OFFLINE (CHECK WI-FI)")
                            time.sleep(2.5)
                            continue

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
        now = time.time()
        with self.lock:
            if detected_persons:
                self.cached_persons = detected_persons
                self.last_person_seen_time = now
            else:
                # Grace period: hold previous verified detection for up to 0.7s
                # to prevent momentary dropouts/flickering when user tilts head, speaks, or blinks
                if now - self.last_person_seen_time > 0.7:
                    self.cached_persons = []

            # Multi-frame temporal confirmation for weapons:
            # Prevents single-frame glitches, optical artifacts, or lighting flicker from causing false alerts
            if detected_weapons:
                self.weapon_threat_counter = min(5, self.weapon_threat_counter + 1)
                self.last_weapon_seen_time = now
                if self.weapon_threat_counter >= 2:
                    self.cached_weapons = detected_weapons
            else:
                self.weapon_threat_counter = max(0, self.weapon_threat_counter - 1)
                if self.weapon_threat_counter == 0 or (now - self.last_weapon_seen_time > 0.6):
                    self.cached_weapons = []

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
                weapon_counter = self.weapon_threat_counter

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

            # Weapon confirmation: require at least 2 consecutive detections to eliminate glitches
            is_confirmed_weapon = (len(cached_w) > 0 and weapon_counter >= 2)
            weapon_count = len(cached_w) if is_confirmed_weapon else 0

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
                label = f"ZONE: {zone_mode.upper()} [BREACH!]" if intruder_count > 0 else f"ZONE: {zone_mode.upper()}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                min_x = max(10, min(630 - lw - 12, int(np.min(pts[:, 0]))))
                min_y = max(26, min(470, int(np.min(pts[:, 1])) + 24))
                cv2.rectangle(display_frame, (min_x, min_y - 20), (min_x + lw + 12, min_y), border_color, -1)
                cv2.putText(display_frame, label, (min_x + 6, min_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

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
                cv2.line(display_frame, (x2, y2), (x2 - c_len, y2), box_color, 3, cv2.LINE_AA)

                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(display_frame, (x1, max(15, y1 - 20)), (x1 + tw + 8, max(15, y1)), box_color, -1)
                cv2.putText(display_frame, tag, (x1 + 4, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Render Weapon Bounding Boxes (confirmed threats only)
            if is_confirmed_weapon:
                for w in cached_w:
                    wx1, wy1, wx2, wy2 = w['coords']
                    w_tag = f"WEAPON: {w['name']} {w['conf']*100:.0f}%"
                    w_color = (0, 69, 255)  # Neon Orange-Red
                    cv2.rectangle(display_frame, (wx1, wy1), (wx2, wy2), w_color, 2, cv2.LINE_AA)
                    (wtw, wth), _ = cv2.getTextSize(w_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(display_frame, (wx1, max(15, wy1 - 20)), (wx1 + wtw + 8, max(15, wy1)), w_color, -1)
                    cv2.putText(display_frame, w_tag, (wx1 + 4, max(12, wy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # 4. Flashing Alert Banners & Smart Snapshot Recording
            is_weapon_threat = is_confirmed_weapon
            is_intrusion_threat = intruder_count > 0

            if is_weapon_threat:
                cv2.rectangle(display_frame, (0, 0), (640, 28), (0, 35, 220), -1)
                w_name = cached_w[0]['name'] if cached_w else 'WEAPON'
                cv2.putText(display_frame, f"CRITICAL: {weapon_count} WEAPON DETECTED ({w_name}) - {self.name}",
                            (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

                # Rate-limit weapon incident logging to avoid spamming multiple records for 1 event
                now = time.time()
                if not self.weapon_incident_active and (now - self.last_weapon_incident_time > 10.0):
                    self.weapon_incident_active = True
                    self.last_weapon_incident_time = now
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
                            details=f"Confirmed weapon threat ({w_name}) detected on camera #{self.camera_id} with {cached_w[0]['conf']*100:.1f}% confidence."
                        )
                    except Exception as err:
                        print(f"[CameraProcessor {self.camera_id}] Error logging weapon incident:", err)
            else:
                self.weapon_incident_active = False

            if is_intrusion_threat and not is_weapon_threat:
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
                # Fast pre-flight check for HTTP / RTSP streams (2.0s timeout)
                import urllib.parse
                import socket
                parsed = urllib.parse.urlparse(source)
                if parsed.hostname:
                    port = parsed.port or (80 if parsed.scheme == 'http' else (443 if parsed.scheme == 'https' else 554))
                    try:
                        sock = socket.create_connection((parsed.hostname, port), timeout=2.0)
                        sock.close()
                    except (socket.timeout, TimeoutError):
                        return {
                            'success': False,
                            'error': f"Connection timed out reaching {parsed.hostname}:{port}. Check that the phone is on the same Wi-Fi network and 'Start server' is running in IP Webcam."
                        }
                    except ConnectionRefusedError:
                        return {
                            'success': False,
                            'error': f"Connection refused by {parsed.hostname}:{port}. The phone is reachable, but IP Webcam server is not started. Open the app, scroll to the bottom, and tap 'Start server'."
                        }
                    except Exception as net_err:
                        return {
                            'success': False,
                            'error': f"Cannot connect to {parsed.hostname}:{port}: {net_err}"
                        }

                cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return {
                    'success': False,
                    'error': 'Could not open video stream. Check IP address, port, and stream path (e.g. /video).'
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
