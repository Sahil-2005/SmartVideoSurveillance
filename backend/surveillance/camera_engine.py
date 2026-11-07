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


def is_complete_person(x1, y1, x2, y2, conf, frame=None):
    """
    Validates whether a detected bounding box represents a genuine human person,
    calibrated for webcam users, seated postures, and room occupants, while strictly
    rejecting inanimate objects (folded laundry/clothes on beds, plush toys, cushions,
    pillows, bags) and isolated waving hands.

    Validation Criteria:
    1. Calibrated Confidence Gate: conf >= 0.50.
       - Rejects low-confidence inanimate proposals (clothes piles, blankets, chairs)
         which typically score 0.35-0.48.
    2. Physical Dimensions & Scale (in 640x480 resolution):
       - Allows chest-level webcam crops and background occupants: height >= 75px, width >= 45px, area >= 6,000px^2.
    3. Small-Candidate Precision Gate:
       - If area < 9,000px^2, requires conf >= 0.65 to reject small plush toys, dolls, and stray objects.
    4. Anatomical Human Aspect Ratio:
       - Close-up webcam chest crop (w >= 220px or area >= 40,000px^2):
         Human shoulders fill the frame horizontally, so 0.55 <= aspect <= 4.20.
       - Mid-range / distant occupants (w < 220px and area < 40,000px^2):
         Human beings (standing, crouching, or seated) are ALWAYS vertically oriented (aspect >= 1.08).
         This strictly eliminates horizontal folded laundry (aspect ~0.93), cushions, pillows, and bags.
    5. Anti-Hand & Anti-Limb Rejection:
       - Eliminates isolated hands/arms (area < 8,500px^2 with narrow width < 50px or extreme aspect ratios).
    """
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False

    # 1. Minimum confidence threshold (rejects low-confidence inanimate objects)
    if conf < 0.50:
        return False

    area = w * h
    aspect = h / float(w)

    # 2. Minimum physical dimensions (in 640x480 resolution)
    if h < 75 or w < 45 or area < 6000:
        return False

    # 3. Small-scale confidence gate
    if area < 9000 and conf < 0.65:
        return False

    # 4. Anatomical Human Aspect Ratio
    if w >= 220 or area >= 40000:
        if aspect < 0.55 or aspect > 4.20:
            return False
    else:
        if aspect < 1.08 or aspect > 4.20:
            return False

    # 5. Anti-Hand & Anti-Limb Rejection
    if area < 8500 and (w < 50 or aspect > 3.5 or aspect < 0.85):
        return False

    return True


def filter_and_suppress_persons(raw_candidates, frame=None):
    """
    Applies complete person verification and Non-Maximum Suppression (NMS) /
    Containment deduplication. Eliminates duplicate boxes on the same person,
    preserves distinct nearby individuals, and rejects false positive limbs.
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
                # Suppress only if significant IoU (> 0.45) or if an inner candidate
                # is mostly contained inside a larger box (> 0.75)
                if iou > 0.45 or containment > 0.75:
                    is_duplicate = True
                    break

        if not is_duplicate:
            final_persons.append(cand)

    return final_persons


def filter_and_suppress_weapons(raw_weapons, detected_persons=None):
    """
    Validates weapon candidates (Knife & Pistol) and applies NMS deduplication.
    
    Validation Criteria:
    1. Physical Handgun Geometry (PISTOL / GUN):
       - Handguns are compact hand-held firearms:
         - Area: 800px^2 <= area <= 11,000px^2 (strictly rejects furniture cushions, doorframes, and background patches).
         - Max dimension <= 120px, Aspect ratio <= 1.85 (rejects tall vertical slivers and wide flat objects).
         - Confidence >= 0.40 when near person, >= 0.65 for standalone firearms.
    2. Physical Bladed Weapon Geometry (KNIFE):
       - Knives are elongated bladed tools (aspect ratio 1.30 to 8.5, area 600px^2 to 45,000px^2).
       - When held near/by person: conf >= 0.20.
       - Standalone knives away from people: conf >= 0.55.
    3. Person Arm-Reach Spatial Association:
       - Confirms candidate is within physical arm's reach of a person (dx <= 110px, dy <= 80px).
       - Rejects ceiling objects (fans/vents) high above human head (wy2 < py1 - 25).
    4. Overlap & Conflict Resolution:
       - If an elongated knife candidate overlaps with a compact pistol candidate, the knife takes precedence.
    5. Non-Maximum Suppression (IoU > 0.35).
    """
    valid_candidates = []
    for w in raw_weapons:
        wx1, wy1, wx2, wy2 = w['coords']
        conf = w['conf']
        name = (w.get('name') or w.get('class') or '').upper()
        ww = wx2 - wx1
        wh = wy2 - wy1
        area = ww * wh
        max_dim = max(ww, wh)
        min_dim = max(1, min(ww, wh))
        aspect = max_dim / float(min_dim)

        if ww < 16 or wh < 16:
            continue

        # Proximity to person check (arm's reach: horizontal <= 110px, vertical <= 80px)
        near_person = False
        is_above_person = False
        body_overlap = False
        if detected_persons:
            for p in detected_persons:
                px1, py1, px2, py2 = p['coords'] if isinstance(p, dict) else p
                overlap_x = max(0, min(wx2, px2) - max(wx1, px1))
                if (wx2 >= px1 - 110 and wx1 <= px2 + 110 and wy2 >= py1 - 80 and wy1 <= py2 + 80):
                    # Check if candidate is high above the person's head on ceiling
                    if wy2 < py1 - 25:
                        is_above_person = True
                        break
                    # If candidate extends above head height (wy1 < py1 - 35),
                    # it must horizontally overlap the body (held upright in hand above shoulders)
                    if wy1 < py1 - 35 and overlap_x == 0:
                        is_above_person = True
                        break
                    near_person = True
                    if overlap_x > 0:
                        body_overlap = True
                    break

        if is_above_person:
            continue

        is_knife = ('KNIFE' in name)

        if not is_knife:
            # PISTOL / GUN validation: compact firearm geometry
            if area < 800 or area > 11000:
                continue
            if max_dim > 120:
                continue
            if aspect > 1.85:
                continue
            min_conf = 0.40 if near_person else 0.65
            if conf < min_conf:
                continue
        else:
            # KNIFE validation: handheld elongated bladed weapon
            # Limits area <= 40,000px^2 (allows close-up hand+blade while rejecting full-screen stripes)
            if area < 600 or area > 40000:
                continue
            if aspect < 1.30 or aspect > 8.5:
                continue
            # Held near and overlapping body: conf >= 0.20. Standalone: conf >= 0.55
            min_conf = 0.20 if (near_person and body_overlap) else (0.35 if near_person else 0.55)
            if conf < min_conf:
                continue

        valid_candidates.append(w)

    if not valid_candidates:
        return []

    # Sort descending by confidence
    valid_candidates.sort(key=lambda x: x['conf'], reverse=True)

    # NMS Deduplication with Knife precedence on elongated shapes
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


def is_combustion_color(crop):
    """
    Verifies that a candidate fire bounding box contains genuine combustion flame incandescence
    rather than cold daylight, fluorescent lamps, or passive orange/red fabric (clothing, blankets,
    bags, cushions).
    
    Combustion Flame Criteria:
    - Genuine combustion flames are self-luminous and exhibit high red dominance:
      mean(R) >= 135 and (mean(R) - mean(B)) >= 35.
    - OR localized bright incandescent flame core:
      at least 20% of pixels in the bounding box satisfy (R > 175, R > B + 40, R > G * 0.90).
    """
    if crop is None or crop.size == 0:
        return False
    b, g, r = cv2.split(crop)
    mean_r = float(np.mean(r))
    mean_b = float(np.mean(b))
    mean_g = float(np.mean(g))

    # Real combustion fire has high red dominance over blue (R - B >= 35) and high mean R (>= 135)
    r_b_diff = mean_r - mean_b
    if mean_r >= 135 and r_b_diff >= 35:
        return True

    # Check for concentrated incandescent flame core (bright fiery pixels)
    # Fire core: R > 175, R > B + 40, R > G * 0.90
    warm_core = (r > 175) & (r > b + 40) & (r > g * 0.90)
    warm_pct = np.count_nonzero(warm_core) / float(crop.shape[0] * crop.shape[1])
    return warm_pct >= 0.20


def is_human_head_false_smoke(smoke_box, detected_persons):
    """
    Checks if a candidate smoke detection is physically sitting directly on top of
    a detected human's shoulders/torso, representing hair/head misclassification.
    """
    if not detected_persons:
        return False

    sx1, sy1, sx2, sy2 = smoke_box
    scx = (sx1 + sx2) / 2.0

    for p in detected_persons:
        px1, py1, px2, py2 = p['coords'] if isinstance(p, dict) else p
        pcx = (px1 + px2) / 2.0
        pw = px2 - px1
        ph = py2 - py1

        # Horizontally aligned with human center
        if abs(scx - pcx) < 0.40 * pw:
            # Sits directly on top of shoulders or overlaps upper torso / head
            if (sy2 >= py1 - 35) and (sy1 <= py1 + 0.35 * ph) and (sy1 >= py1 - 0.75 * ph):
                return True

    return False


def filter_and_suppress_fire_smoke(raw_candidates, detected_persons=None, frame=None):
    """
    Validates fire and smoke candidates and applies NMS deduplication.
    
    Validation Criteria:
    1. Combustion Flame (FIRE):
       - Confidence >= 0.35
       - Dimensional bounds: 400px^2 <= area <= 260,000px^2 (rejects sub-pixel glints)
       - Color spectrum check: verifies warm combustion incandescence, rejecting cold daylight / office lamps.
    2. Smoke Dispersion (SMOKE):
       - Confidence >= 0.45 for standalone smoke, >= 0.32 if accompanying confirmed fire.
       - Area >= 1,000px^2 (smoke is a diffuse volumetric phenomenon).
       - Anti-head/hair filter: rejects smoke candidates aligned directly above a human torso as hair.
    3. Cross-Hazard Conflict Resolution:
       - When a candidate triggers both FIRE and SMOKE in the same region, FIRE takes precedence if combustion colors are present.
    4. Non-Maximum Suppression (IoU > 0.35).
    """
    valid_fire = []
    raw_smoke = []
    h_f, w_f = (frame.shape[:2]) if frame is not None else (480, 640)

    for c in raw_candidates:
        x1, y1, x2, y2 = c['coords']
        conf = c['conf']
        name = (c.get('name') or c.get('class') or '').upper()
        w = x2 - x1
        h = y2 - y1
        area = w * h
        if w < 16 or h < 16 or area < 400 or area > 260000:
            continue
        if 'FIRE' in name:
            if conf < 0.45:
                continue
            if frame is not None:
                crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                if not is_combustion_color(crop):
                    continue
            valid_fire.append(c)
        else:
            raw_smoke.append(c)

    has_fire = len(valid_fire) > 0
    valid_smoke = []
    for c in raw_smoke:
        x1, y1, x2, y2 = c['coords']
        conf = c['conf']
        w = x2 - x1
        h = y2 - y1
        area = w * h
        if area < 1000:
            continue
        min_conf = 0.32 if has_fire else 0.45
        if conf < min_conf:
            continue
        if is_human_head_false_smoke(c['coords'], detected_persons):
            continue
        valid_smoke.append(c)

    candidates = valid_fire + valid_smoke
    if not candidates:
        return []

    candidates.sort(key=lambda x: x['conf'], reverse=True)
    final_hazards = []
    for cand in candidates:
        cx1, cy1, cx2, cy2 = cand['coords']
        cand_area = max(1, (cx2 - cx1) * (cy2 - cy1))
        is_dup = False
        for acc in final_hazards:
            ax1, ay1, ax2, ay2 = acc['coords']
            acc_area = max(1, (ax2 - ax1) * (ay2 - ay1))
            ix1, iy1 = max(cx1, ax1), max(cy1, ay1)
            ix2, iy2 = min(cx2, ax2), min(cy2, ay2)
            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h
            if inter_area > 0:
                iou = inter_area / float(cand_area + acc_area - inter_area)
                if iou > 0.35:
                    is_dup = True
                    break
        if not is_dup:
            final_hazards.append(cand)
    return final_hazards


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
        self.fire_smoke_model = None
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
            fire_smoke_pt = os.path.join(base_dir, 'fire_smoke_yolov8n.pt')

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

            try:
                self.fire_smoke_model = YOLO(fire_smoke_pt if os.path.exists(fire_smoke_pt) else 'fire_smoke_yolov8n.pt')
            except Exception as e:
                print(f"[InferenceWorker] Error loading fire/smoke model: {e}")
                self.fire_smoke_model = None

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

            detected_persons = None
            detected_weapons = None
            detected_fire_smoke = None

            # 1. Person Detection (classes=[0] for person) with Complete Person Filtering & NMS
            if self.person_model and target_proc.person_detection_enabled:
                try:
                    p_results = self.person_model(frame, classes=[0], conf=0.45, imgsz=480, verbose=False)
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
                    detected_persons = []

            # Determine interleaved auxiliary model schedule
            # Guaranteed: on any live tick, AT MOST 1 auxiliary model runs!
            is_static = target_proc.is_static_image
            cycle = self._inference_counter % 2
            
            should_run_weapon = (
                self.weapon_model and 
                target_proc.weapon_detection_enabled and 
                (is_static or (cycle == 0 and (detected_persons or (self._inference_counter % 4 == 0))))
            )
            should_run_fire = (
                self.fire_smoke_model and 
                target_proc.fire_smoke_detection_enabled and 
                (is_static or (cycle == 1))
            )

            # 2. Weapon Detection (Knife & Pistol) - Dual-Stage Global + Person-ROI Inference
            if should_run_weapon:
                try:
                    raw_weapons = []
                    # Stage A: Global Frame Inference (imgsz=480, classes=[0, 3] for Gun & Knife)
                    w_results = self.weapon_model(frame, conf=0.18, classes=[0, 3], imgsz=480, verbose=False)
                    for r in w_results:
                        for box in r.boxes:
                            wx1, wy1, wx2, wy2 = map(int, box.xyxy[0])
                            w_conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            cls_name = 'KNIFE' if cls_id == 3 else 'PISTOL'
                            raw_weapons.append({
                                'coords': (wx1, wy1, wx2, wy2),
                                'conf': w_conf,
                                'name': cls_name
                            })

                    # Stage B: High-Precision Person-ROI Inference
                    # Zooms into person + arm-reach interaction zone to catch narrow bladed knives
                    # and small gripped pistols in high resolution!
                    if detected_persons:
                        h_f, w_f = frame.shape[:2]
                        for p in detected_persons:
                            px1, py1, px2, py2 = p['coords']
                            pad_x, pad_y = 80, 50
                            cx1 = max(0, px1 - pad_x)
                            cy1 = max(0, py1 - pad_y)
                            cx2 = min(w_f, px2 + pad_x)
                            cy2 = min(h_f, py2 + pad_y)
                            crop = frame[cy1:cy2, cx1:cx2]
                            if crop.shape[0] >= 32 and crop.shape[1] >= 32:
                                roi_res = self.weapon_model(crop, conf=0.18, classes=[0, 3], imgsz=480, verbose=False)
                                for r in roi_res:
                                    for box in r.boxes:
                                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                                        w_conf = float(box.conf[0])
                                        cls_id = int(box.cls[0])
                                        cls_name = 'KNIFE' if cls_id == 3 else 'PISTOL'
                                        raw_weapons.append({
                                            'coords': (bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1),
                                            'conf': w_conf,
                                            'name': cls_name
                                        })

                    detected_weapons = filter_and_suppress_weapons(raw_weapons, detected_persons or [])
                except Exception as e:
                    print(f"[InferenceWorker] Weapon detection error on Cam {target_proc.camera_id}: {e}")
                    detected_weapons = []

            # 3. Fire & Smoke Detection (D-Fire YOLOv8 Nano)
            if should_run_fire:
                try:
                    fs_results = self.fire_smoke_model(frame, conf=0.25, imgsz=480, verbose=False)
                    raw_fs = []
                    for r in fs_results:
                        for box in r.boxes:
                            fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                            f_conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            cls_name = self.fire_smoke_model.names.get(cls_id, 'fire').upper()
                            raw_fs.append({
                                'coords': (fx1, fy1, fx2, fy2),
                                'conf': f_conf,
                                'name': cls_name
                            })
                    detected_fire_smoke = filter_and_suppress_fire_smoke(raw_fs, detected_persons or target_proc.cached_persons, frame)
                except Exception as e:
                    print(f"[InferenceWorker] Fire/smoke detection error on Cam {target_proc.camera_id}: {e}")
                    detected_fire_smoke = []

            # Feed detection results back to camera processor
            target_proc.receive_inference_results(detected_persons, detected_weapons, detected_fire_smoke, frame_idx)


class CameraProcessor:
    """
    Independent capture and visual rendering pipeline for a single camera.
    Decoupled from AI inference and Django web request threads.
    """

    def __init__(self, camera_id, name, stream_url, camera_type='ip_webcam', location='',
                 zone_mode='right', zone_config=None, person_detection=True, weapon_detection=True,
                 fire_smoke_detection=True):
        self.camera_id = camera_id
        self.name = name
        self.stream_url = str(stream_url).strip()
        self.camera_type = camera_type
        self.location = location

        self.zone_mode = zone_mode
        self.zone_config = zone_config or {'x1': 320, 'y1': 0, 'x2': 640, 'y2': 480}
        self.person_detection_enabled = person_detection
        self.weapon_detection_enabled = weapon_detection
        self.fire_smoke_detection_enabled = fire_smoke_detection

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
        self.is_static_image = False

        # Detection cache & counts
        self.cached_persons = []
        self.cached_weapons = []
        self.cached_fire_smoke = []
        self.last_person_seen_time = 0
        self.person_count = 0
        self.intruder_count = 0
        self.weapon_count = 0
        self.fire_smoke_count = 0

        # Intrusion temporal confirmation & incident debounce
        self.intruder_threat_counter = 0
        self.last_intruder_seen_time = 0

        # Weapon temporal confirmation & incident debounce
        self.weapon_threat_counter = 0
        self.last_weapon_seen_time = 0
        self.weapon_incident_active = False
        self.last_weapon_incident_time = 0

        # Fire & Smoke temporal confirmation & incident debounce
        self.fire_smoke_threat_counter = 0
        self.last_fire_smoke_seen_time = 0
        self.fire_incident_active = False
        self.last_fire_incident_time = 0

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
                        zone_config=None, person_detection=None, weapon_detection=None,
                        fire_smoke_detection=None):
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
            if fire_smoke_detection is not None:
                self.fire_smoke_detection_enabled = fire_smoke_detection

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
                self.is_static_image = True
                img = cv2.imread(source)
                if img is not None:
                    with self.lock:
                        self.latest_raw_frame = img
                        self.raw_frame_id += 1
                        self.status = "ONLINE"
                        self.last_seen = datetime.now()
                    while self.running and not self._need_reconnect:
                        time.sleep(1.0)
                        with self.lock:
                            # Keep frame fresh for worker if inference hasn't run yet
                            if self.last_inferred_frame_id < 0 or (self.raw_frame_id == self.last_inferred_frame_id):
                                self.raw_frame_id += 1
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

    def receive_inference_results(self, detected_persons, detected_weapons, detected_fire_smoke, frame_idx):
        """Called by SharedInferenceWorker with fresh detection results."""
        now = time.time()
        with self.lock:
            if detected_persons is not None:
                if detected_persons:
                    self.cached_persons = detected_persons
                    self.last_person_seen_time = now
                else:
                    if not self.is_static_image and (now - self.last_person_seen_time > 0.7):
                        self.cached_persons = []
            else:
                if not self.is_static_image and (now - self.last_person_seen_time > 1.2):
                    self.cached_persons = []

            # Multi-frame temporal confirmation for weapons
            if detected_weapons is not None:
                if detected_weapons:
                    self.weapon_threat_counter = min(5, self.weapon_threat_counter + 1)
                    self.last_weapon_seen_time = now
                    if self.is_static_image or self.weapon_threat_counter >= 2:
                        self.cached_weapons = detected_weapons
                else:
                    if not self.is_static_image:
                        self.weapon_threat_counter = max(0, self.weapon_threat_counter - 1)
                        if self.weapon_threat_counter == 0 or (now - self.last_weapon_seen_time > 0.6):
                            self.cached_weapons = []
            else:
                # Interleaved cycle where weapon was not run: preserve verified threats up to 0.8s
                if not self.is_static_image and (now - self.last_weapon_seen_time > 0.8):
                    self.cached_weapons = []

            # Multi-frame temporal confirmation for Fire & Smoke
            if detected_fire_smoke is not None:
                if detected_fire_smoke:
                    self.fire_smoke_threat_counter = min(5, self.fire_smoke_threat_counter + 1)
                    self.last_fire_smoke_seen_time = now
                    if self.is_static_image or self.fire_smoke_threat_counter >= 2:
                        self.cached_fire_smoke = detected_fire_smoke
                else:
                    if not self.is_static_image:
                        self.fire_smoke_threat_counter = max(0, self.fire_smoke_threat_counter - 1)
                        if self.fire_smoke_threat_counter == 0 or (now - self.last_fire_smoke_seen_time > 0.8):
                            self.cached_fire_smoke = []
            else:
                # Interleaved cycle where fire/smoke was not run: preserve verified hazards up to 1.2s
                if not self.is_static_image and (now - self.last_fire_smoke_seen_time > 1.2):
                    self.cached_fire_smoke = []

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
                cached_fs = list(self.cached_fire_smoke)
                weapon_counter = self.weapon_threat_counter
                fs_counter = self.fire_smoke_threat_counter
                is_static = self.is_static_image

            if raw_frame is None or status in ("OFFLINE", "ERROR", "DISCONNECTED"):
                time.sleep(0.05)
                continue

            display_frame = cv2.resize(raw_frame, (640, 480))
            zone_polygon = get_zone_polygon(zone_mode, zone_config)

            person_count = len(cached_p)
            raw_intruders = []

            for p in cached_p:
                x1, y1, x2, y2 = p['coords']
                conf = p['conf']
                pw = x2 - x1
                ph = y2 - y1
                p_area = pw * ph
                is_in_zone, anchor_pt = is_person_in_zone(x1, y1, x2, y2, zone_polygon)
                # An unauthorized intrusion requires a confirmed person candidate (conf >= 0.52 and area >= 6,500px^2)
                if is_in_zone and conf >= 0.52 and p_area >= 6500:
                    raw_intruders.append({
                        'coords': (x1, y1, x2, y2),
                        'conf': conf,
                        'anchor': anchor_pt
                    })

            # Intrusion temporal confirmation: require 2 detections on video, immediate on static images
            now = time.time()
            if raw_intruders:
                self.intruder_threat_counter = min(5, self.intruder_threat_counter + 1)
                self.last_intruder_seen_time = now
            else:
                if not is_static:
                    self.intruder_threat_counter = max(0, self.intruder_threat_counter - 1)

            is_confirmed_intruder = (len(raw_intruders) > 0 and (is_static or self.intruder_threat_counter >= 2))
            intruder_count = len(raw_intruders) if is_confirmed_intruder else 0

            detected_persons_with_zone = []
            for p in cached_p:
                x1, y1, x2, y2 = p['coords']
                conf = p['conf']
                pw = x2 - x1
                ph = y2 - y1
                p_area = pw * ph
                is_in_zone, anchor_pt = is_person_in_zone(x1, y1, x2, y2, zone_polygon)
                is_breach = is_confirmed_intruder and is_in_zone and conf >= 0.52 and p_area >= 6500
                detected_persons_with_zone.append({
                    'coords': (x1, y1, x2, y2),
                    'conf': conf,
                    'anchor': anchor_pt,
                    'is_intruder': is_breach
                })

            # Weapon confirmation: require 2 detections on video, immediate on static images
            is_confirmed_weapon = (len(cached_w) > 0 and (is_static or weapon_counter >= 2))
            weapon_count = len(cached_w) if is_confirmed_weapon else 0

            # Fire & Smoke confirmation
            is_confirmed_fs = (len(cached_fs) > 0 and (is_static or fs_counter >= 2))
            fire_smoke_count = len(cached_fs) if is_confirmed_fs else 0
            fire_hazards = [h for h in cached_fs if 'FIRE' in h['name']] if is_confirmed_fs else []
            smoke_hazards = [h for h in cached_fs if 'SMOKE' in h['name']] if is_confirmed_fs else []
            fire_count = len(fire_hazards)
            smoke_count = len(smoke_hazards)

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

            # 4. Render Fire & Smoke Bounding Boxes
            if is_confirmed_fs:
                for h in cached_fs:
                    hx1, hy1, hx2, hy2 = h['coords']
                    h_conf = h['conf']
                    h_name = h['name']
                    if 'FIRE' in h_name:
                        h_color = (0, 69, 255)  # Flaming Orange/Red
                        h_tag = f"FIRE {h_conf*100:.0f}%"
                    else:
                        h_color = (180, 160, 130)  # Smoke Gray/Khaki
                        h_tag = f"SMOKE {h_conf*100:.0f}%"

                    cv2.rectangle(display_frame, (hx1, hy1), (hx2, hy2), h_color, 2, cv2.LINE_AA)
                    (htw, hth), _ = cv2.getTextSize(h_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(display_frame, (hx1, max(15, hy1 - 20)), (hx1 + htw + 8, max(15, hy1)), h_color, -1)
                    cv2.putText(display_frame, h_tag, (hx1 + 4, max(12, hy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # 5. Flashing Alert Banners & Smart Snapshot Recording
            is_fire_threat = fire_count > 0
            is_smoke_threat = smoke_count > 0 and not is_fire_threat
            is_weapon_threat = is_confirmed_weapon
            is_intrusion_threat = intruder_count > 0

            # Priority 1: Fire & Smoke Hazard Alerts
            if is_fire_threat:
                cv2.rectangle(display_frame, (0, 0), (640, 28), (0, 40, 230), -1)
                cv2.putText(display_frame, f"CRITICAL: FIRE HAZARD DETECTED - {self.name}",
                            (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

                now = time.time()
                if not self.fire_incident_active and (now - self.last_fire_incident_time > 10.0):
                    self.fire_incident_active = True
                    self.last_fire_incident_time = now
                    try:
                        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        evidence_dir = os.path.join(base_dir, 'media', 'evidence')
                        os.makedirs(evidence_dir, exist_ok=True)
                        snap_name = f"cam{self.camera_id}_fire_{int(time.time())}.jpg"
                        snap_path = os.path.join(evidence_dir, snap_name)
                        cv2.imwrite(snap_path, display_frame)

                        cam_obj = Camera.objects.filter(id=self.camera_id).first()
                        SecurityIncident.objects.create(
                            camera=cam_obj,
                            threat_type="FIRE DETECTED",
                            severity='CRITICAL',
                            source=f"{self.name} ({self.location or 'Hazard Sector'})",
                            intruder_count=fire_count,
                            snapshot=f"evidence/{snap_name}",
                            details=f"Active combustion fire detected on camera #{self.camera_id} with {fire_hazards[0]['conf']*100:.1f}% confidence."
                        )
                    except Exception as err:
                        print(f"[CameraProcessor {self.camera_id}] Error logging fire incident:", err)
            elif is_smoke_threat:
                cv2.rectangle(display_frame, (0, 0), (640, 28), (50, 75, 100), -1)
                cv2.putText(display_frame, f"WARNING: SMOKE HAZARD DETECTED - {self.name}",
                            (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

                now = time.time()
                if not self.fire_incident_active and (now - self.last_fire_incident_time > 10.0):
                    self.fire_incident_active = True
                    self.last_fire_incident_time = now
                    try:
                        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        evidence_dir = os.path.join(base_dir, 'media', 'evidence')
                        os.makedirs(evidence_dir, exist_ok=True)
                        snap_name = f"cam{self.camera_id}_smoke_{int(time.time())}.jpg"
                        snap_path = os.path.join(evidence_dir, snap_name)
                        cv2.imwrite(snap_path, display_frame)

                        cam_obj = Camera.objects.filter(id=self.camera_id).first()
                        SecurityIncident.objects.create(
                            camera=cam_obj,
                            threat_type="SMOKE DETECTED",
                            severity='CRITICAL',
                            source=f"{self.name} ({self.location or 'Hazard Sector'})",
                            intruder_count=smoke_count,
                            snapshot=f"evidence/{snap_name}",
                            details=f"Smoke dispersion hazard detected on camera #{self.camera_id} with {smoke_hazards[0]['conf']*100:.1f}% confidence."
                        )
                    except Exception as err:
                        print(f"[CameraProcessor {self.camera_id}] Error logging smoke incident:", err)
            else:
                self.fire_incident_active = False

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
            hud_text = f"CAM {self.camera_id}: {self.name}  |  PEOPLE: {person_count}  |  INTRUDERS: {intruder_count}  |  WEAPONS: {weapon_count}  |  HAZARDS: {fire_smoke_count}"
            cv2.putText(display_frame, hud_text, (10, 473), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

            # Update public state
            with self.lock:
                self.person_count = person_count
                self.intruder_count = intruder_count
                self.weapon_count = weapon_count
                self.fire_smoke_count = fire_smoke_count
                if fire_count > 0:
                    self.status = "FIRE ALERT"
                elif smoke_count > 0:
                    self.status = "SMOKE ALERT"
                elif weapon_count > 0:
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
                'fire_smoke_count': self.fire_smoke_count,
                'alert_active': (self.intruder_count > 0 or self.weapon_count > 0 or self.fire_smoke_count > 0),
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

            active_cameras = Camera.objects.filter(is_active=True).exclude(camera_type='upload')
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
                        weapon_detection=cam.weapon_detection,
                        fire_smoke_detection=cam.fire_smoke_detection
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
                        weapon_detection=cam.weapon_detection,
                        fire_smoke_detection=cam.fire_smoke_detection
                    )

            # Stop processors for deactivated or deleted cameras
            to_remove = [cid for cid, proc in self.processors.items() if cid not in active_ids and proc.camera_type != 'upload']
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
                    weapon_detection=cam.weapon_detection,
                    fire_smoke_detection=cam.fire_smoke_detection
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
                weapon_detection=camera_obj.weapon_detection,
                fire_smoke_detection=camera_obj.fire_smoke_detection
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

    def get_all_stats(self, exclude_upload=True):
        """Aggregate detection statistics across all active cameras."""
        with self.lock:
            if not self._synced:
                self.sync_from_db()
            procs = list(self.processors.values())

        total_persons = 0
        total_intruders = 0
        total_weapons = 0
        total_fire_smoke = 0
        online_count = 0
        camera_stats = []

        for p in procs:
            if exclude_upload and p.camera_type == 'upload':
                continue
            stats = p.get_stats()
            camera_stats.append(stats)
            if "ONLINE" in stats['status'] or "ALERT" in stats['status'] or "MONITORING" in stats['status']:
                online_count += 1
            total_persons += stats['person_count']
            total_intruders += stats['intruder_count']
            total_weapons += stats['weapon_count']
            total_fire_smoke += stats.get('fire_smoke_count', 0)

        if total_fire_smoke > 0:
            system_status = f"CRITICAL: {total_fire_smoke} FIRE/SMOKE HAZARD(S) DETECTED"
        elif total_weapons > 0:
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
            'total_cameras': len(camera_stats),
            'online_cameras': online_count,
            'total_persons': total_persons,
            'total_intruders': total_intruders,
            'total_weapons': total_weapons,
            'total_fire_smoke': total_fire_smoke,
            'alert_active': (total_intruders > 0 or total_weapons > 0 or total_fire_smoke > 0),
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
