import cv2
import numpy as np
from ultralytics import YOLO

p_model = YOLO('backend/yolo11n.pt')
w_model = YOLO('backend/weapon_yolov8n.pt')

test_images = {
    'Real Pistol': 'backend/media/check.jpg',
    'Real Knife': 'backend/media/check2.jpg',
    'User Room (False Intruder)': 'backend/media/focused_feed_false_intruder.jpg',
    'User Hand Raised': r'C:\Users\sahil\.gemini\antigravity\brain\688465fe-413d-4545-809d-4f156e304f50\.user_uploaded\media_1793715439530_8d27d359.png',
    'User in front': r'C:\Users\sahil\.gemini\antigravity\brain\688465fe-413d-4545-809d-4f156e304f50\.user_uploaded\media_1793715531761_6409bb7f.jpg',
    'User Intruder screen': r'C:\Users\sahil\.gemini\antigravity\brain\688465fe-413d-4545-809d-4f156e304f50\.user_uploaded\media_1793717073123_cebf3a9c.png',
}

for label, path in test_images.items():
    img = cv2.imread(path)
    if img is None:
        continue
    img = cv2.resize(img, (640, 480))
    
    # 1. Person detection
    p_res = p_model(img, classes=[0], conf=0.40, imgsz=480, verbose=False)
    persons = []
    for r in p_res:
        for b in r.boxes:
            persons.append({'coords': [int(x) for x in b.xyxy[0]], 'conf': float(b.conf[0])})
            
    # 2. Weapon detection
    w_res = w_model(img, conf=0.40, imgsz=480, verbose=False)
    raw_w = []
    for r in w_res:
        for b in r.boxes:
            raw_w.append({
                'coords': [int(x) for x in b.xyxy[0]],
                'conf': float(b.conf[0]),
                'name': w_model.names[int(b.cls[0])].upper()
            })
            
    filtered = []
    for w in raw_w:
        wx1, wy1, wx2, wy2 = w['coords']
        conf = w['conf']
        ww = wx2 - wx1
        wh = wy2 - wy1
        area = ww * wh
        
        print(f"DEBUG {label}: w={w['name']}, conf={conf:.3f}, ww={ww}, wh={wh}, area={area}, persons={len(persons)}")
        
        if ww < 20 or wh < 20 or area < 500 or area > 75000:
            print(f"DEBUG {label}: skipped by bounds")
            continue
            
        aspect = max(ww, wh) / max(1, min(ww, wh))
        if aspect > 7.0:
            print(f"DEBUG {label}: skipped by aspect")
            continue
            
        if persons:
            near_person = any(
                (wx2 >= px[0] - 130 and wx1 <= px[2] + 130 and wy2 >= px[1] - 130 and wy1 <= px[3] + 130)
                for px in [p['coords'] for p in persons]
            )
            print(f"DEBUG {label}: near_person={near_person}")
            if near_person and conf < 0.45:
                print(f"DEBUG {label}: skipped by conf near person (< 0.45)")
                continue
            if not near_person and conf < 0.70:
                print(f"DEBUG {label}: skipped by conf far person (< 0.70)")
                continue
        else:
            print(f"DEBUG {label}: no persons, conf={conf}")
            if conf < 0.70:
                print(f"DEBUG {label}: skipped by standalone conf (< 0.70)")
                continue
                
        filtered.append(w)
        
    print(f"[{label}] Raw weapons: {len(raw_w)}, Filtered: {len(filtered)}")
    for fw in filtered:
        c = fw['coords']
        name = fw['name']
        conf = fw['conf']
        print(f"   -> {name} conf={conf:.3f} coords={c}")
