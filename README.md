# Smart Video Surveillance & Threat Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.x-green.svg)](https://www.djangoproject.com/)
[![YOLO](https://img.shields.io/badge/YOLO-v11n%20%7C%20v8n-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-ready, AI-driven Smart CCTV Surveillance and Threat Detection Platform designed to monitor live video feeds (webcams, RTSP/IP cameras) and uploaded media. Built on top of **Django**, **OpenCV**, and **Ultralytics YOLO (YOLO11n + custom YOLOv8n)**, the system delivers high-precision person detection, restricted zone intrusion detection, weapon threat identification (knife & pistol), real-time evidence snapshot logging, and interactive security telemetry.

---

## Key Capabilities & Features

### 1. Accurate Person Detection & Complete-Person Geometry
- **Complete Body Geometry Filtering:** Evaluates candidate bounding boxes using aspect ratio ($0.60 \le \text{aspect} \le 4.0$), physical dimensions ($h \ge 110\text{px}, w \ge 65\text{px}$), and minimum pixel volume ($\text{area} \ge 8,000\text{px}^2$).
- **Anti-Limb & Hand Suppression:** Distinguishes between complete human torsos/occupants and isolated arms or hands. Raising a hand or waving does not register as a person or inflate headcounts.
- **Deduplication via NMS & Containment:** Eliminates duplicate bounding boxes for the same individual ($\text{IoU} > 0.30$ or nested containment $> 0.45$).
- **Temporal Grace Period ($0.7\text{s}$):** Retains verified person state across momentary head tilts, blinks, or speaking pauses to eliminate visual strobe-flickering.

### 2. Restricted Zone & Intrusion Detection
- **Preset Security Sectors:** Instant zone coverage including **Right Sector**, **Left Sector**, **Center Box**, and **Corridor**.
- **Interactive Drag-to-Draw Custom Boundary:** Security operators can draw arbitrary polygonal/rectangular intrusion boundaries directly over the live video feed with dynamic dashed amber preview, automated coordinate translation to native camera resolution ($640\times 480$), and instant API persistence (`/api/cameras/<id>/set-zone`).
- **Anatomical Anchor & Volumetric Overlap Verification:** Intrusion breaches are evaluated using physical body anchor points (head, torso centroid, base, and mid-height lateral flanks) requiring $\ge 25\%$ volumetric overlap. Grazing background corners or empty bounding box margins never trigger false alarms.

### 3. Weapon & Critical Threat Detection (Knife & Pistol)
- **Dual-Model Inference:** Leverages `weapon_yolov8n.pt` running at native $640\times 640$ resolution alongside `yolo11n.pt`.
- **Arm-Reach Spatial Association ($\le 110\text{px}$):** Confirms proximity to a detected person within physical reaching distance ($\le 110\text{px}$), linking weapons directly to wielders while rejecting phantom weapons on distant walls, ceilings, or curtains.
- **Class-Calibrated Sensitivity & Physical Geometry:**
  - **Knives ($\text{conf} \ge 0.35$):** Optimized for bladed weapons held within arm's reach ($\le 120\text{px}$) with aspect ratio $\le 7.0$ and area $\le 45,000\text{px}^2$. Standalone knives require $\text{conf} \ge 0.65$.
  - **Pistols ($\text{conf} \ge 0.73$):** Enforces compact palm-sized handgun constraints ($\text{area} \le 7,000\text{px}^2$, $\text{max\_dimension} \le 95\text{px}$, $\text{aspect} \le 1.7$). This strictly eliminates large furniture cushions ($\text{area} > 12,000\text{px}^2$) and human heads/hair silhouettes ($\text{area} > 10,000\text{px}^2$) from ever triggering false pistol alarms.
- **Multi-Frame Temporal Confirmation ($\ge 2$ Consecutive Cycles):** Prevents transient optical noise, hand flicks, or lighting shifts from triggering critical alarms.
- **Incident Debouncing ($10.0\text{s}$ Cooldown):** Prevents repetitive alert flooding for ongoing single events.

### 4. Real-Time Evidence Snapshotting & Incident Management
- **Automatic Evidence Snapshots:** Threat states automatically record high-resolution annotated snapshots to `media/evidence/`.
- **Database-Backed Incidents:** Logs severity levels (`CRITICAL`, `WARNING`), threat types, camera origins, timestamps, and confidence telemetry into `SecurityIncident` database models.
- **Evidence Lightbox Viewer:** Inspect snapshot evidence in high-resolution directly from the dashboard or incident logs without page navigation.

### 5. Multi-Camera Scalability & Shared Inference Architecture
- **Multi-Stream Support:** Seamlessly handles local webcams, network RTSP streams, IP Webcam streams, and recorded media files.
- **Shared AI Inference Worker:** Decouples video capture and rendering from neural network execution. A single synchronized worker schedules round-robin inference across active streams, guaranteeing predictable CPU utilization and preventing thread contention.
- **Low-Latency Streaming:** Drop-stale buffer architecture for network streams ensures zero latency accumulation.

### 6. Modern Glassmorphic Operations Dashboard
- **Live Stream Views:** Toggle between multi-feed **Grid View** and high-detail **Focus View**.
- **Live Telemetry & Metrics:** Real-time metrics for total active cameras, live people counts, and active threat indicators.
- **Chart.js Threat Analytics:** Live 7-day threat trend line chart and event breakdown donut chart.
- **Asynchronous Telemetry Polling:** Polls `/api/stats` at $1\text{s}$ intervals without page reloads.

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend Framework** | Python 3.9+, Django 5.x |
| **Computer Vision & AI** | OpenCV 5.x (`cv2`), PyTorch, Ultralytics YOLO11n (Person) & YOLOv8n (Weapon) |
| **Frontend** | HTML5, Vanilla CSS3 (Glassmorphism), Vanilla ES6 JavaScript, Canvas API |
| **Data Visualization** | Chart.js 4.x, FontAwesome 6 |
| **Database** | SQLite (Development) / PostgreSQL compatible |
| **Streaming Protocol** | Multipart MJPEG over HTTP (`StreamingHttpResponse`) |

---

## System Architecture (Under the Hood)

```mermaid
flowchart TD
    subgraph Capture & Rendering Layer
        Cam1[Camera 1 Capture Thread] --> Proc1[CameraProcessor 1]
        Cam2[Camera 2 Capture Thread] --> Proc2[CameraProcessor 2]
        Proc1 --> Render1[Visual Render Loop @ 25 FPS]
        Proc2 --> Render2[Visual Render Loop @ 25 FPS]
    end

    subgraph AI Inference Pipeline
        Proc1 -.-> Worker[SharedInferenceWorker]
        Proc2 -.-> Worker
        Worker --> YOLOPerson[YOLO11n Person Detection]
        Worker --> YOLOWeapon[YOLOv8n Weapon Detection]
        YOLOPerson --> PersonFilter[Complete Person Geometry & NMS Filter]
        YOLOWeapon --> WeaponFilter[Class & Proximity Weapon Filter]
        PersonFilter --> Sync[Sync Results via Lock]
        WeaponFilter --> Sync
        Sync -.-> Proc1
        Sync -.-> Proc2
    end

    subgraph Output & Incident Handling
        Render1 --> Stream[MJPEG Stream /video_feed/]
        Render1 --> Snap[Evidence Snapshot media/evidence/]
        Render1 --> DB[(SQLite SecurityIncident)]
        Stream --> Browser[Client Dashboard Web UI]
        DB --> Browser
    end
```

1. **`CameraProcessor` (Decoupled Capture & Rendering):**
   Each camera operates its own capture thread and independent visual rendering thread running at $\sim 25\text{ FPS}$. Network streams continually flush stale buffers to maintain sub-second latency.
2. **`SharedInferenceWorker` (Synchronized Round-Robin AI):**
   A singleton worker thread cycles through active cameras using a round-robin schedule. Exactly one inference executes at any instant on the CPU, preventing thread starvation and thermal throttling.
3. **Contextual Filter Engine:**
   Raw candidate boxes are validated through [`filter_and_suppress_persons`](file:///e:/Freelance/SmartVideoSurveillance/backend/surveillance/camera_engine.py#L169-L220) and [`filter_and_suppress_weapons`](file:///e:/Freelance/SmartVideoSurveillance/backend/surveillance/camera_engine.py#L223-L280) before being cached in the processor state.
4. **State Machine & Incident Persistence:**
   When a threat is confirmed over consecutive frames, `CameraProcessor` writes a snapshot evidence frame to disk, records a `SecurityIncident`, and triggers the client-side alarm.

---

## Installation & Setup

### 1. Clone Repository & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Sahil-2005/SmartVideoSurveillance.git
cd SmartVideoSurveillance

# Navigate to backend
cd backend

# Create virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Create virtual environment (Linux / macOS)
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Apply Migrations & Verify Setup

```bash
python manage.py migrate
python manage.py check
```

### 4. Launch Development Server

```bash
python manage.py runserver
```

Open your browser and navigate to: `http://127.0.0.1:8000/`

---

## Default Administrative Credentials

- **Username:** `admin`
- **Password:** `admin123`

---

## API Endpoints Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/video_feed/` | `GET` | MJPEG video stream of the primary active camera. |
| `/video_feed/<camera_id>/` | `GET` | Direct MJPEG video stream for a specific camera ID. |
| `/api/stats` | `GET` | Aggregated surveillance telemetry (people, threats, active alerts, camera statuses). |
| `/api/analytics` | `GET` | Incident statistics, 7-day distribution, hourly charts, and sector breakdowns. |
| `/api/cameras/<id>/set-zone` | `POST` | Updates zone mode (`none`, `preset`, `custom`) and polygon coordinate configuration. |
| `/api/cameras/<id>/toggle-detection`| `POST` | Toggles person detection or weapon detection on/off per camera. |
| `/upload_source/` | `POST` | Uploads recorded video/image files for dedicated offline AI analysis. |
| `/clear_upload/` | `POST` | Clears and shuts down uploaded media streams. |

---

## Interactive Zone Drawing Guide

1. Navigate to the **Dashboard** and switch the primary camera tile to **Focus View**.
2. Click the **✏️ Draw Zone** button beneath the video player.
3. Click and drag across the video feed to draw your custom restricted perimeter.
4. Release the mouse button — the coordinates will automatically normalize to native camera resolution and save to the backend.
5. Any person stepping inside or reaching into the drawn boundary will trigger an **UNAUTHORIZED INTRUSION** alert.

---

## Author

<div align="left">
  <a href="https://github.com/Sahil-2005" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
  <a href="https://www.linkedin.com/in/sahil-gawade-920a0a242/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  <a href="mailto:gawadesahil.dev@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail" />
  </a>
  <a href="https://leetcode.com/u/sahilgawade4321/" target="_blank">
    <img src="https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=LeetCode&logoColor=black" alt="LeetCode" />
  </a>
  <a href="https://sahil-gawade.vercel.app/" target="_blank">
    <img src="https://img.shields.io/badge/Portfolio-2563EB?style=for-the-badge&logo=Vercel&logoColor=white" alt="Portfolio" />
  </a>
</div>
