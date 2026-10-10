# Smart Video Surveillance & Threat Detection System

A web-based Smart Video Surveillance and Threat Detection System designed to monitor live video feeds or uploaded media. It utilizes computer vision and deep learning (YOLO11n) to automatically detect persons, generating alerts and providing a beautiful administrative dashboard for monitoring.

## Features
- **Live Webcam Monitoring:** Connects to the system's primary webcam for real-time person detection.
- **Media Upload:** Allows users to upload recorded videos or images for real-time inference and analysis.
- **AI Engine (YOLO11 Nano):** Uses the state-of-the-art, CPU-optimized YOLO11n model via the Ultralytics framework.
- **Dynamic Glassmorphic UI:** A highly polished, modern, dark-themed administrative dashboard to monitor operations.
- **Authentication System:** Secure login access for administrators.
- **Mock Event Logs:** UI setup for tracking detection and system events.

---

## Prerequisites
- Python 3.9+
- A working webcam (optional, but required for Live Webcam feed functionality)

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd SmartVideoSurveillance
   ```

2. **Navigate to the Backend directory:**
   ```bash
   cd backend
   ```

3. **Create and activate a virtual environment:**
   - **Windows:**
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the database migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Start the Development Server:**
   ```bash
   python manage.py runserver
   ```
   *(Note: The very first time you start the server, it will automatically download the `yolo11n.pt` model weights, which is ~5.4MB).*

---

## Usage
1. Open your browser and go to: `http://127.0.0.1:8000/`
2. You will be redirected to the login page.
3. Log in using the default administrator credentials:
   - **Username:** `admin`
   - **Password:** `admin123`
4. From the sidebar, you can navigate between:
   - **Live Webcam:** Turn on/off your webcam and watch live detections.
   - **Media Upload:** Upload an image or video to run detection on it.
   - **Logs:** View the system activity logs.

---

## Technology Stack
- **Backend:** Python, Django
- **Computer Vision / AI:** OpenCV, PyTorch, Ultralytics YOLO11
- **Frontend:** HTML5, Vanilla CSS3 (Glassmorphism design), Vanilla JavaScript
- **Database:** SQLite (Development)

---

## How It Works (Under the Hood)

1. **The Backend (Django & OpenCV):** 
   - We use Django as the core server. It exposes an endpoint (`/video_feed`) that utilizes a `StreamingHttpResponse` to serve a multipart video stream directly to the browser.
   - A background generator loop using OpenCV (`cv2`) continually reads frames from the active source (either the live webcam or an uploaded media file).
2. **The AI Engine (YOLO11):**
   - Every frame captured by OpenCV is passed through the pre-trained `YOLO11n` (Nano) object detection model.
   - We optimized it to run on a standard CPU by resizing frames (640x480) and filtering the AI's predictions to only detect class `0` (which corresponds to a 'person' in the COCO dataset).
   - Once a person is detected, OpenCV draws precise bounding boxes directly onto the image frame before sending it to the user.
3. **The Global State & Concurrency:**
   - Because video streaming is continuous and blocking, we use Python's `threading.Lock()` and global variables to safely pass active data (like how many people are detected and the current camera status) between the OpenCV processing loop and Django's API endpoints.
4. **The Frontend (Polling & UI):**
   - The frontend is built using standard HTML/CSS with a custom, premium Glassmorphic design. 
   - To keep the architecture lightweight and avoid complex WebSocket setups (like Redis/Daphne), we use JavaScript `fetch()` polling. Every 1 second, the frontend silently asks the `/api/stats` endpoint for the latest detection data (person count, system status) and seamlessly updates the UI without ever reloading the page.

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
