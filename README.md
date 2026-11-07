# Smart Video Surveillance & Threat Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.x-green.svg)](https://www.djangoproject.com/)
[![YOLO](https://img.shields.io/badge/YOLO-v11n%20%7C%20v8n%20%7C%20D--Fire-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-ready, AI-driven Smart CCTV Surveillance, Threat Detection, and Hazard Warning Platform engineered for real-time monitoring across live video feeds (webcams, RTSP/IP cameras) and forensic media uploads. Built on top of **Django**, **OpenCV**, and **Ultralytics YOLO (YOLO11n, custom weapon YOLOv8n, and fine-tuned D-Fire YOLOv8n)**, the system delivers high-precision person tracking, custom restricted zone intrusion boundaries, armed weapon identification (knife & pistol), real-time fire and smoke combustion hazard alerting, automated evidence snapshot logging, and interactive security telemetry.

Engineered specifically with **CPU-first interleaved scheduling**, allowing the entire multi-model AI suite to operate seamlessly on standard low-power laptops (e.g. Intel Core i3 dual-core CPUs with integrated graphics) with sub-second latency and zero thermal throttling.

---

## Key Capabilities & Features

### 1. High-Precision Person Detection & Adaptive Human Geometry
- **Calibrated Confidence Gate ($\text{conf} \ge 0.50$):** Eliminates low-confidence inanimate proposals (folded laundry piles on beds, chairs with coats, hanging towels, and bags) which typically score $0.35 - 0.48$.
- **Morphological Human Aspect Ratio & Tapering:**
  - **Close-Up Webcam Occupants ($w \ge 220\text{px}$ or $\text{area} \ge 40,000\text{px}^2$):** Broad torso/chest crops spanning the frame allow $0.55 \le \text{aspect} \le 4.20$.
  - **Mid-Range & Distant Room Occupants ($w < 220\text{px}$ and $\text{area} < 40,000\text{px}^2$):** Humans (standing, seated, or crouching) are strictly vertically oriented ($\text{aspect} \ge 1.08$). This physically rejects squarish/horizontal lumps like piles of folded clothes ($\text{aspect} \approx 0.93$), pillows, blankets, and bags.
- **Small-Scale Precision Gating:** Objects smaller than $9,000\text{px}^2$ must meet high confidence ($\text{conf} \ge 0.65$), preventing small plush toys, shoes, or shelf items from registering as people.
- **Anti-Limb & Hand Rejection:** Strictly eliminates isolated waving hands or arms ($\text{area} < 8,500\text{px}^2$ with $w < 50\text{px}$ or extreme aspect ratios). Raising a hand or gesturing never inflates headcounts or registers as an occupant.
- **Deduplication via NMS & Containment:** Eliminates duplicate bounding boxes for the same individual ($\text{IoU} > 0.45$ or nested containment $> 0.75$), while preserving adjacent distinct individuals.
- **Temporal Grace Period ($0.7\text{s}$):** Retains verified person state across momentary head tilts, blinks, or speaking pauses to eliminate visual strobe-flickering.

### 2. Restricted Zone & Intrusion Detection
- **Preset Security Sectors:** Instant zone coverage including **Right Sector**, **Left Sector**, **Center Box**, and **Corridor**.
- **Interactive Drag-to-Draw Custom Boundary:** Security operators can draw arbitrary polygonal/rectangular intrusion boundaries directly over the live video feed with dynamic dashed amber preview, automated coordinate translation to native camera resolution ($640\times 480$), and instant API persistence (`/api/cameras/<id>/set-zone`).
- **Anatomical Anchor & Volumetric Overlap Verification:** Intrusion breaches are evaluated using physical body anchor points (head, torso centroid, base, and mid-height lateral flanks) requiring $\ge 25\%$ volumetric overlap. Grazing background corners or empty bounding box margins never trigger false alarms.
- **Intruder Gating & Multi-Frame Temporal Confirmation:**
  - Breaching occupants must be verified human candidates ($\text{conf} \ge 0.52, \text{area} \ge 6,500\text{px}^2$).
  - Requires $\ge 2$ consecutive frames in the restricted zone on live feeds before declaring an active intrusion breach. Transient lighting glints or single-frame flickers never trigger sirens or emergency banners.

### 3. Armed Weapon & Threat Classification (Knife & Pistol)
- **Dual-Stage Person-ROI Inference Pipeline:**
  - **Stage A (Global Frame Scan):** Rapid global inference ($480\text{px}$) detects macroscopic firearm displays and unholstered weapons.
  - **Stage B (High-Resolution Zoom ROI):** Dynamically crops a padded bounding region ($+80\text{px}$ horizontal, $+50\text{px}$ vertical) around each detected person at native resolution ($480\text{px}$). Catches narrow bladed knives, angled kitchen knives, and concealed small handguns that downsampling degrades.
- **Physical Handgun Geometry Filtering (Pistol / Gun):**
  - Handgun area strictly constrained: $800\text{px}^2 \le \text{area} \le 11,000\text{px}^2$, $\text{max\_dimension} \le 120\text{px}$, and $\text{aspect} \le 1.85$.
  - This eliminates false alarms on floral cushions ($22,500\text{px}^2$), dark doorframes, ceiling beams, and human heads/neck shadows.
  - Confidence threshold: $\ge 0.40$ when held near a person, $\ge 0.65$ for standalone handguns.
- **Elongated Bladed Tool Geometry (Knife):**
  - Knives exhibit elongated slender geometry: $1.30 \le \text{aspect} \le 8.5$, $600\text{px}^2 \le \text{area} \le 40,000\text{px}^2$.
  - **Overhead Fixture & Clothesline Rejection:** Any candidate extending high above head level ($wy_1 < py_1 - 35\text{px}$) must horizontally overlap the occupant's body, strictly rejecting floating doorway headers, hanging clotheslines, and curtain edges.
  - **Calibrated Proximity Thresholds:** Handheld knives physically overlapping the occupant body require $\text{conf} \ge 0.20$; non-overlapping adjacent candidates require $\text{conf} \ge 0.35$; standalone bladed objects require $\text{conf} \ge 0.55$.
  - Conflict resolution: Overlapping blade-grip proposals prioritize knife classification over pistol grips.
- **Overhead Fixture Rejection:** Candidates positioned entirely above human head height ($y_2 < py_1 - 25\text{px}$) are rejected as ceiling fixtures or ventilation ducts.
- **Multi-Frame Temporal Confirmation ($\ge 2$ Consecutive Cycles):** Prevents transient optical noise or rapid gestures from triggering critical alarms.
- **Incident Debouncing ($10.0\text{s}$ Cooldown):** Prevents repetitive alert flooding for ongoing single events.

### 4. Fire & Smoke Hazard Detection (D-Fire YOLOv8 Nano)
- **Ultra-Lightweight Neural Architecture:** Fine-tuned D-Fire YOLOv8 Nano model weights (`fire_smoke_yolov8n.pt`, $3.2\text{M}$ parameters, $\sim 6.2\text{ MB}$) running at $480\times 480$ resolution for maximum CPU efficiency ($\sim 45\text{ms}$ inference latency).
- **Dual-Hazard Classification:** Independently detects and differentiates between active combustion flames (`FIRE`) and atmospheric smoke plumes/haze (`SMOKE`).
- **Incandescent Combustion Verification (Fire vs Orange Fabrics):**
  - Calibrated confidence gate ($\text{conf} \ge 0.45$) with spatial bounding ($400\text{px}^2 \le \text{area} \le 260,000\text{px}^2$).
  - Evaluates color channel distribution: verifies genuine self-luminous flame incandescence ($\text{mean}(R) \ge 135$ and $\text{mean}(R) - \text{mean}(B) \ge 35$, or $\ge 20\%$ concentrated flame core pixels with $R > 175, R > B + 40, R > 0.90 \cdot G$). Passive orange/red fabrics (t-shirts, blankets, cushions, towels) and cold daylight glare are strictly rejected.
- **Anti-Head / Anti-Hair False Smoke Suppression:**
  - Evaluates smoke candidates against detected persons. If a candidate is horizontally aligned with a human torso ($|\Delta cx| < 0.40 \cdot pw$) and rests directly on shoulders or head height without accompanying flame, it is rejected as head/hair misclassification.
  - Standalone smoke requires $\text{conf} \ge 0.45$ and $\text{area} \ge 1,000\text{px}^2$; smoke accompanying confirmed fire triggers at $\text{conf} \ge 0.32$.
- **Hazard-Specific Visual Overlays & Warnings:** Displays glowing flame-red bounding boxes for fire and khaki/slate gray boxes for smoke, accompanied by prominent top banner alerts: `CRITICAL: FIRE HAZARD DETECTED` and `WARNING: SMOKE HAZARD DETECTED`.
- **Dual-Tone Synthesizer Siren:** Dispatches audio alarms via Web Audio API oscillators specifically modulated for combustion emergencies.
- **Automated Evidence Logging:** Instantaneously records snapshot frames (`cam<id>_fire_<ts>.jpg` / `cam<id>_smoke_<ts>.jpg`) directly into `SecurityIncident` audit logs.

### 5. Low-Resource Hardware Optimization (Dual-Core CPU Friendly)
Designed to run at consistent high frame rates without overheating on modest hardware:
- **Target Baseline Hardware:** Intel Core i3-1005G1 (2 cores, 4 threads @ 1.20GHz), Intel UHD Graphics (128MB VRAM), 12GB RAM.
- **Strictly Interleaved Auxiliary Scheduling:**
  - Person Detection runs every tick at $480\text{px}$ ($\sim 35\text{ms}$).
  - Auxiliary Cycle 0: Weapon Detection ($640\text{px}$) runs only on even ticks when an occupant is present (or every 4th tick when unoccupied).
  - Auxiliary Cycle 1: Fire & Smoke Detection ($480\text{px}$) runs on odd ticks.
  - **Single Auxiliary Guarantee:** At most ONE auxiliary model executes on any single frame tick. Total compute time per inference pass stays strictly under $90\text{ms}$, capping dual-core CPU utilization below $35\%$.
- **State-Preserving Grace Windows:** When an auxiliary model is interleaved, previous confirmed threat detections are preserved for up to $1.2\text{s}$, completely eliminating visual box flickering while yielding 50% CPU savings.
- **Instantaneous Forensic Uploads:** Media files and uploaded static images immediately run all three models synchronously on ingestion for zero-wait inspection.

### 6. Real-Time Evidence Snapshotting & Incident Audit
- **Automatic Evidence Snapshots:** Threat states automatically record high-resolution annotated snapshots to `media/evidence/`.
- **Database-Backed Incidents:** Logs severity levels (`CRITICAL`, `WARNING`), threat types, camera origins, timestamps, and confidence telemetry into `SecurityIncident` database models.
- **Evidence Lightbox Viewer:** Inspect snapshot evidence in high-resolution directly from the dashboard or incident logs without page navigation.
- **CSV Audit Export & History Purging:** One-click CSV audit report generation and log clearing with file cleanup.

### 7. Multi-Camera Scalability & Shared AI Worker
- **Multi-Stream Support:** Seamlessly handles local webcams, network RTSP streams, IP Webcam streams, and recorded media files.
- **Shared AI Inference Worker:** Decouples video capture and rendering from neural network execution. A single synchronized worker schedules round-robin inference across active streams, guaranteeing predictable CPU utilization and preventing thread contention.
- **Low-Latency Streaming:** Drop-stale buffer architecture for network streams ensures zero latency accumulation.

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend Framework** | Python 3.9+, Django 5.x |
| **Computer Vision & AI** | OpenCV 5.x (`cv2`), PyTorch, Ultralytics YOLO11n (Person), YOLOv8n (Weapon), YOLOv8n (Fire/Smoke) |
| **Frontend** | HTML5, Vanilla CSS3 (Glassmorphism), Vanilla ES6 JavaScript, Canvas API |
| **Audio Synthesizer** | Web Audio API (Multi-frequency Dual-Tone Threat Sirens) |
| **Data Visualization** | Chart.js 4.x, FontAwesome 6 |
| **Database** | SQLite (Development) / PostgreSQL compatible |
| **Streaming Protocol** | Multipart MJPEG over HTTP (`StreamingHttpResponse`) |

---

## System Architecture

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
        Worker --> YOLOPerson[YOLO11n Person Detection @ 480px]
        Worker -. Cycle 0 .-> YOLOWeapon[Dual-Stage YOLOv8n Weapon: Global + Person-ROI @ 480px]
        Worker -. Cycle 1 .-> YOLOFire[D-Fire YOLOv8n Fire & Smoke @ 480px]
        YOLOPerson --> PersonFilter[Adaptive Scale, Anti-Limb & NMS Filter]
        YOLOWeapon --> WeaponFilter[Physical Handgun/Blade Geometry & Overlap Filter]
        YOLOFire --> FireFilter[Combustion Spectrum & Anti-Head Smoke Filter]
        PersonFilter --> Sync[Sync Results via Lock]
        WeaponFilter --> Sync
        FireFilter --> Sync
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
| `/api/stats` | `GET` | Aggregated surveillance telemetry (people, weapons, fire/smoke, active alerts, camera statuses). |
| `/api/cameras` | `GET` | Lists all configured camera nodes with feature flags and live metrics. |
| `/api/cameras/add` | `POST` | Registers a new camera node with custom detection flags (`person_detection`, `weapon_detection`, `fire_smoke_detection`). |
| `/api/cameras/<id>/edit` | `POST` | Updates camera settings, endpoints, or active AI detection models. |
| `/api/cameras/<id>/toggle` | `POST` | Enables or disables an active camera stream. |
| `/api/cameras/<id>/set-zone` | `POST` | Updates zone mode (`none`, `preset`, `custom`) and polygon coordinate configuration. |
| `/upload_source/` | `POST` | Uploads recorded video/image files for dedicated offline AI forensic analysis. |
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
