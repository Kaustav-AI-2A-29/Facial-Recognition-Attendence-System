# Smart Facial Recognition Attendance System
## Setup & Usage Guide

---

## Project Structure

```
project/
├── dataset/
│   ├── Alice.jpg          ← One clear, front-facing photo per person
│   ├── Bob.jpg
│   └── Charlie.jpg
│
├── face_attendance.py     ← Main program
├── requirements.txt       ← Python dependencies
├── attendance.csv         ← Auto-created on first run
└── SETUP.md               ← This file
```

---

## Step-by-Step Installation

### 1 — Install Python 3.10 or later
Download from https://www.python.org/downloads/  
During installation, tick **"Add Python to PATH"**.

### 2 — (Windows only) Install build tools for dlib
`face_recognition` requires `dlib`, which needs a C++ compiler.

**Option A – Easiest (pre-built wheel)**
```bash
pip install cmake
pip install dlib
```

**Option B – Install Visual Studio Build Tools**  
Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/  
Select "Desktop development with C++" workload.

### 3 — Install all Python dependencies
Open a terminal in VS Code (`Ctrl + ~`) and run:
```bash
pip install -r requirements.txt
```

### 4 — Add face images to the `dataset/` folder
- Name each image file after the person: `Alice.jpg`, `Bob.jpg`, etc.
- Use a **clear, front-facing** photo with good lighting.
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### 5 — Run the program
```bash
python face_attendance.py
```

Press **Q** to quit the webcam window.

---

## Configuration Options
Edit the constants near the top of `face_attendance.py`:

| Constant        | Default | Description                                          |
|-----------------|---------|------------------------------------------------------|
| `DATASET_DIR`   | `dataset` | Folder containing reference images                 |
| `ATTENDANCE_FILE` | `attendance.csv` | Output CSV path                         |
| `TOLERANCE`     | `0.50`  | Match strictness — lower = stricter (range 0.4–0.6)  |
| `FRAME_SCALE`   | `0.25`  | Downscale factor — smaller = faster but less accurate|
| `PROCESS_EVERY_N` | `2`   | Process every Nth frame (higher = less CPU load)     |

---

## Output — attendance.csv
```
Name,Date,Time
Alice,2026-04-04,09:15:22
Bob,2026-04-04,09:18:10
Charlie,2026-04-04,09:20:41
```
- One entry per person per calendar day.
- Duplicate entries are prevented both in-memory and via CSV check.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'face_recognition'` | Run `pip install face_recognition` |
| `dlib` fails to install on Windows | Install cmake first, then dlib |
| Webcam not detected | Check another app can use the camera; try index `1` in `cv2.VideoCapture(1)` |
| Face not recognised | Lower `TOLERANCE` to 0.45; ensure dataset image is clear |
| Slow performance | Increase `PROCESS_EVERY_N` to 3 or 4 |

---

## How It Works — Architecture Overview

```
Webcam Frame
    │
    ▼
Resize to 25%  ──────────────────────────────┐
    │                                         │ (skipped frames
    ▼                                         │  reuse cached data)
BGR → RGB conversion                         │
    │                                         │
    ▼                                         │
face_recognition.face_locations()            │
    │                                         │
    ▼                                         │
face_recognition.face_encodings()            │
    │                                         │
    ▼                                         │
compare_faces()  +  face_distance()          │
    │                                         │
    ▼                                         │
Best match below tolerance?                  │
   YES → label = Name                        │
   NO  → label = "Unknown"                   │
    │                                         │
    ▼                                         │
mark_attendance() — write CSV if new         │
    │                                         │
    ▼                                         │
Scale coords back to full size ◄─────────────┘
    │
    ▼
Draw box + name + confidence on frame
    │
    ▼
cv2.imshow() → Display
```
