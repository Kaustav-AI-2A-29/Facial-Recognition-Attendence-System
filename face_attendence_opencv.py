"""
============================================================
  Smart Facial Recognition Attendance System
  OpenCV LBPH Edition  —  NO dlib required
  Author   : AI/ML Engineer (Claude)
  Run with : python face_attendence_opencv.py

  Requirements
  ─────────────
  pip install opencv-contrib-python   ← NOT opencv-python
  pip install numpy pandas

  Why LBPH instead of hash matching?
  ────────────────────────────────────
  Perceptual hash (the old approach) compares raw pixel patterns.
  It cannot distinguish between two people and is trivially fooled
  by lighting changes, angle, or textured backgrounds.

  LBPH (Local Binary Pattern Histograms) is OpenCV's purpose-built
  face recognition algorithm. It:
    • Encodes local texture patterns around each pixel
    • Is robust to lighting variation
    • Returns a real distance score (lower = closer match)
    • Trains in seconds on a small dataset
    • Runs entirely on CPU with no GPU or dlib needed

  Bug fixes over previous version
  ─────────────────────────────────
  BUG 1 — Misidentification
    Old: hash + histogram similarity (not face recognition)
    Fix: LBPH recognizer trained on reference face crops

  BUG 2 — Two people marked when one is present
    Old: Confirmation buffer reset on ANY background detection
    Fix: Position-based buffer (face_id) tracks each detected
         face independently — background noise cannot reset it

  BUG 3 — Background objects detected as faces
    Old: Haar cascade with weak parameters + no size/shape filter
    Fix: minNeighbors=8 + MIN_FACE_SIZE=80px + aspect ratio gate
         + CLAHE normalisation before detection
============================================================
"""

import cv2
import numpy as np
import pandas as pd
import os
import csv
import logging
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("attendance_opencv")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATASET_DIR    = "dataset"
ATTENDANCE_FILE = "attendance.csv"

# LBPH recognizer settings
# Lower confidence = better match. 0 = perfect, 100+ = poor match.
# Tune here if recognition is too strict or too loose:
LBPH_THRESHOLD = 50         # Accept match only if confidence < this value
                              # Try 50 if faces are missed; try 40 if wrong person shown

# Face detection (Haar cascade)
CASCADE_PATH        = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_SCALE          = 1.08   # Smaller = more detections but slower (1.05–1.15)
HAAR_MIN_NEIGHBORS  = 9      # Higher = fewer false positives (6–10)
MIN_FACE_SIZE       = 80     # Pixels — rejects hands, shadows, small objects
MAX_FACE_RATIO      = 1.5    # Max width/height ratio — rejects elongated shapes

# Confirmation: require N consecutive matching frames before marking attendance
CONFIRM_FRAMES  = 6         # At ~15 fps effective = ~530 ms of agreement (more time = less false positives)

# Frame processing
PROCESS_EVERY_N = 2          # Detect on every Nth frame (reduces CPU load)
FRAME_SCALE     = 0.5        # Downsample for detection (0.5 = half size, faster)

# CLAHE lighting normalisation
USE_CLAHE  = True
CLAHE_CLIP = 2.5
CLAHE_TILE = (8, 8)

# Webcam
WEBCAM_INDEX  = 0
WEBCAM_WIDTH  = 1280
WEBCAM_HEIGHT = 720

# Display colours (BGR)
COL_KNOWN   = (0, 210, 100)   # Green  – confirmed identity
COL_UNKNOWN = (0, 60, 220)    # Red    – not recognised
COL_PENDING = (0, 190, 220)   # Amber  – accumulating confirmation frames
COL_HUD     = (200, 200, 200)

# Reference image size fed to LBPH (must be consistent)
FACE_SIZE = (200, 200)


# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — CHECK opencv-contrib IS INSTALLED
# ═══════════════════════════════════════════════════════════════

def _check_contrib():
    """
    Raise a clear error if opencv-contrib-python is not installed.
    The cv2.face module (which contains LBPHFaceRecognizer) is ONLY
    available in opencv-contrib-python, not in plain opencv-python.
    """
    if not hasattr(cv2, "face"):
        raise ImportError(
            "\n\n  MISSING: cv2.face module not found.\n"
            "  You need opencv-contrib-python, not opencv-python.\n\n"
            "  Fix:\n"
            "    pip uninstall opencv-python -y\n"
            "    pip install opencv-contrib-python\n"
        )


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — LIGHTING NORMALISATION (CLAHE)
# ═══════════════════════════════════════════════════════════════

def make_clahe():
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)


def apply_clahe_gray(gray: np.ndarray, clahe) -> np.ndarray:
    """Apply CLAHE directly to a grayscale image."""
    return clahe.apply(gray)


# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — DATASET LOADER + LBPH TRAINER
# ═══════════════════════════════════════════════════════════════

def build_lbph_recognizer(dataset_dir: str, cascade: cv2.CascadeClassifier):
    """
    Scan the dataset folder, extract face crops from each reference image,
    and train an LBPH recognizer.

    Returns
    -------
    recognizer : cv2.face.LBPHFaceRecognizer  (trained)
    label_map  : dict[int, str]  — maps integer label → person name
    name_map   : dict[str, int]  — reverse: person name → integer label

    Why LBPH?
    ---------
    LBPH divides a face image into a grid of cells. For each pixel it
    computes an 8-bit binary code comparing the pixel to its 8 neighbours.
    The resulting histogram per cell captures local texture patterns that
    are robust to:
      • Uniform lighting changes (monotone transform leaves patterns intact)
      • Small pose / expression differences
      • Mild occlusion

    Training is instant (milliseconds) and prediction runs at >30 fps on CPU.
    LBPH confidence is a chi-square distance — 0 = perfect match, 100+ = poor.
    We accept a match only when confidence < LBPH_THRESHOLD (default 70).

    Folder layouts supported
    -------------------------
    A) Flat   : dataset/Alice.jpg  dataset/Bob.jpg
    B) Nested : dataset/Alice/img1.jpg  dataset/Alice/img2.jpg  …
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    root = Path(dataset_dir)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder '{dataset_dir}' not found. "
            "Create it and add face images."
        )

    # Build {name: [Path, …]}
    person_files: dict[str, list] = {}
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            imgs = [f for f in sub.iterdir() if f.suffix.lower() in supported]
            if imgs:
                person_files[sub.name] = imgs
    if not person_files:
        for f in sorted(root.iterdir()):
            if f.is_file() and f.suffix.lower() in supported:
                person_files.setdefault(f.stem, []).append(f)

    if not person_files:
        raise ValueError(
            f"No face images found in '{dataset_dir}'. "
            "Add .jpg/.png files or sub-folders named after each person."
        )

    clahe = make_clahe()
    label_map: dict[int, str] = {}
    name_map:  dict[str, int] = {}
    faces_train: list[np.ndarray] = []
    labels_train: list[int]       = []

    log.info("Training LBPH recognizer from '%s' …", dataset_dir)
    label_id = 0

    for name, paths in person_files.items():
        label_map[label_id] = name
        name_map[name]      = label_id
        face_count = 0

        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                log.warning("  Cannot read '%s' – skipped.", p.name)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = apply_clahe_gray(gray, clahe)  # Normalise lighting

            # Try to auto-detect and crop the face in the reference image
            detected = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(50, 50)
            )

            if len(detected) > 0:
                # Use the largest detected face
                (x, y, w, h) = max(detected, key=lambda r: r[2] * r[3])
                face_crop = gray[y:y+h, x:x+w]
            else:
                # No face found in reference image — use the whole image.
                # This is a fallback; ideally every reference image has a
                # clearly visible face.
                log.warning(
                    "  No face detected in '%s/%s' — using full image as fallback.",
                    name, p.name
                )
                face_crop = gray

            # Normalise to consistent size
            face_crop = cv2.resize(face_crop, FACE_SIZE)
            faces_train.append(face_crop)
            labels_train.append(label_id)
            face_count += 1

        if face_count == 0:
            log.warning("  No usable images for '%s' – person skipped.", name)
            del label_map[label_id]
            del name_map[name]
        else:
            log.info("  ✔  %-20s  %d image(s) → LBPH trained", name, face_count)
            label_id += 1

    if not faces_train:
        raise ValueError(
            "No face images could be loaded for training. "
            "Ensure dataset images contain clear, front-facing faces."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces_train, np.array(labels_train))
    log.info("LBPH training complete — %d person(s) enrolled.\n", len(label_map))
    return recognizer, label_map, name_map


# ═══════════════════════════════════════════════════════════════
#  SECTION 4 — FACE DETECTION WITH FALSE-POSITIVE FILTERING
# ═══════════════════════════════════════════════════════════════

def detect_faces(gray: np.ndarray,
                 cascade: cv2.CascadeClassifier) -> list[tuple]:
    """
    Detect faces in a grayscale frame and apply three filters to remove
    false positives (background objects, hands, shadows).

    Filter 1 — minNeighbors=8
        The Haar cascade votes: each candidate window is kept only if at
        least N overlapping windows also detect a face there. Higher N means
        fewer spurious detections. 8 is strict; use 6 on dim cameras.

    Filter 2 — minimum size (MIN_FACE_SIZE)
        Rejects detections smaller than 80×80 pixels. Background objects
        and distant incidental detections are typically <40 px.

    Filter 3 — aspect ratio (MAX_FACE_RATIO)
        A human face is roughly square. We reject detections where
        width/height or height/width exceeds 1.5, which eliminates
        elongated shapes (books, screens, door frames).

    Returns list of (x, y, w, h) tuples for faces that pass all filters.
    """
    raw = cascade.detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(raw) == 0:
        return []

    valid = []
    for (x, y, w, h) in raw:
        ratio = max(w, h) / max(min(w, h), 1)
        if ratio <= MAX_FACE_RATIO:
            valid.append((x, y, w, h))

    return valid


# ═══════════════════════════════════════════════════════════════
#  SECTION 5 — N-FRAME CONFIRMATION BUFFER
# ═══════════════════════════════════════════════════════════════

class ConfirmationBuffer:
    """
    Require CONFIRM_FRAMES consecutive identical labels before accepting.

    Keyed by face_id (the index of the face in the current frame's
    detection list), NOT by name.  This is the critical fix over the
    old implementation:

    OLD (broken): buffer.reset() when ANY "Unknown" face appeared.
    → One background detection cleared the whole buffer for real faces.

    NEW (correct): each face position has its own independent history.
    → Background detections don't interfere with the real face's buffer.
    → Buffer entries for positions that disappear are cleaned up via
      clear_stale(), not on-detection.

    State transitions
    -----------------
    pending  : 1 ≤ hits < CONFIRM_FRAMES  (amber box shown)
    confirmed: hits == CONFIRM_FRAMES, all same label (green box)
    unknown  : no consistent label in window (red box)
    """

    def __init__(self):
        self._hist: dict[int, list[str]] = {}

    def update(self, face_id: int, label: str) -> str | None:
        """
        Append label to the sliding window for face_id.
        Returns the confirmed label if the window is full and unanimous,
        otherwise None.
        """
        h = self._hist.setdefault(face_id, [])
        h.append(label)
        if len(h) > CONFIRM_FRAMES:
            h.pop(0)

        # Confirm only when window is full, all entries agree, and it's a real name
        if len(h) == CONFIRM_FRAMES and len(set(h)) == 1 and h[0] != "Unknown":
            return h[0]
        return None

    def is_pending(self, face_id: int) -> bool:
        """True if we have some known-name hits but haven't confirmed yet."""
        h = self._hist.get(face_id, [])
        known = [x for x in h if x != "Unknown"]
        return 0 < len(known) < CONFIRM_FRAMES

    def clear_stale(self, active_ids: set):
        """Remove history for face positions no longer visible in frame."""
        for k in list(self._hist):
            if k not in active_ids:
                del self._hist[k]


# ═══════════════════════════════════════════════════════════════
#  SECTION 6 — ATTENDANCE CSV WRITER
# ═══════════════════════════════════════════════════════════════

class AttendanceWriter:
    """
    Attendance recorder with two-layer duplicate prevention:
      Layer 1 — in-memory set (session-level, instant check)
      Layer 2 — pandas CSV lookup (survives process restarts)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._marked: set = set()
        self._init_csv()

    def _init_csv(self):
        if not os.path.isfile(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                csv.writer(f).writerow(["Name", "Date", "Time"])
            log.info("Created attendance file: '%s'", self.filepath)

    def _in_csv(self, name: str, today: str) -> bool:
        try:
            df = pd.read_csv(self.filepath)
            return not df[(df["Name"] == name) & (df["Date"] == today)].empty
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return False

    def record(self, name: str) -> bool:
        """
        Write a new attendance row.  Returns True if a new row was written.
        Silently skips if already recorded today.
        """
        now   = datetime.now()
        today = now.strftime("%Y-%m-%d")
        key   = f"{name}|{today}"

        if key in self._marked:
            return False
        self._marked.add(key)

        if self._in_csv(name, today):
            return False

        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow([name, today, now.strftime("%H:%M:%S")])
        log.info("  [MARK] %-20s %s  %s", name, today, now.strftime("%H:%M:%S"))
        return True


# ═══════════════════════════════════════════════════════════════
#  SECTION 7 — DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════

def draw_face_box(frame: np.ndarray,
                  x: int, y: int, w: int, h: int,
                  label: str, confidence: float,
                  color: tuple, pending: bool = False) -> None:
    """
    Draw a bounding box, filled label strip, and LBPH confidence bar.
    confidence is displayed as a percentage (higher = better match,
    because we convert the LBPH distance to a 0–100 scale for display).
    """
    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Filled label strip at the bottom of the box
    strip_h = 46
    cv2.rectangle(frame, (x, y + h - strip_h), (x + w, y + h),
                  color, cv2.FILLED)

    # Name
    cv2.putText(frame, label,
                (x + 6, y + h - strip_h + 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)

    # Confidence score + pending hint
    conf_display = f"{confidence:.0f}% match"
    if pending:
        conf_display += "  [confirming…]"
    cv2.putText(frame, conf_display,
                (x + 6, y + h - 6),
                cv2.FONT_HERSHEY_PLAIN, 0.85,
                (220, 220, 220), 1, cv2.LINE_AA)

    # Confidence bar (full width, 4 px above strip)
    bar_w = int(w * min(confidence / 100.0, 1.0))
    cv2.rectangle(frame,
                  (x, y + h - strip_h - 5),
                  (x + bar_w, y + h - strip_h - 1),
                  color, cv2.FILLED)


def draw_hud(frame: np.ndarray, face_count: int, fps: float) -> None:
    cv2.putText(frame,
                f"Faces: {face_count}  |  {fps:.1f} fps  |  Q = Quit",
                (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.35, COL_HUD, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
#  SECTION 8 — WEBCAM INITIALISATION
# ═══════════════════════════════════════════════════════════════

def open_webcam() -> cv2.VideoCapture:
    """
    Open webcam at WEBCAM_INDEX, with automatic fallback to indices 1–3.
    Sets preferred resolution and minimises internal buffer lag.
    The camera opens immediately when the script runs.
    """
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        for idx in range(1, 4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                log.warning("Webcam index %d failed; using index %d.",
                            WEBCAM_INDEX, idx)
                break
        else:
            raise IOError(
                "No webcam detected. "
                "Check the camera is connected and not used by another app."
            )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Pull freshest frame, reduce lag

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Webcam opened at %dx%d. Press  Q  to quit.\n", w, h)
    return cap


# ═══════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN RECOGNITION LOOP
# ═══════════════════════════════════════════════════════════════

def recognize_faces(recognizer,
                    label_map:  dict[int, str],
                    cascade:    cv2.CascadeClassifier,
                    writer:     AttendanceWriter) -> None:
    """
    Real-time recognition loop.

    Per-frame pipeline
    ------------------
    1.  Grab frame from webcam
    2.  Convert to grayscale + apply CLAHE
    3.  Downsample the gray frame for faster detection
    4.  Detect face bounding boxes (with 3-gate filter)
    5.  For each face: crop, resize to FACE_SIZE, predict with LBPH
    6.  Gate on LBPH_THRESHOLD  (conf < threshold → accept)
    7.  Update N-frame confirmation buffer (per face position)
    8.  Confirmed → AttendanceWriter.record()
    9.  Draw annotations on full-size frame
    10. Display; loop until Q
    """
    cap   = open_webcam()
    clahe = make_clahe() if USE_CLAHE else None
    buf   = ConfirmationBuffer()
    inv   = 1.0 / FRAME_SCALE

    frame_n   = 0
    fps_cnt   = 0
    fps       = 0.0
    fps_tick  = cv2.getTickCount()

    # Cache from last processed frame so skipped frames still draw boxes
    cached_faces:  list = []   # (x, y, w, h) at full resolution
    cached_labels: list = []   # (label, conf_pct, color, pending) per face

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame grab failed – retrying …")
            continue

        frame_n += 1
        fps_cnt += 1

        # Update FPS counter every 30 frames
        if fps_cnt >= 30:
            elapsed  = (cv2.getTickCount() - fps_tick) / cv2.getTickFrequency()
            fps      = fps_cnt / max(elapsed, 1e-6)
            fps_cnt  = 0
            fps_tick = cv2.getTickCount()

        # ── Run detection only on every Nth frame ────────────────────────
        if frame_n % PROCESS_EVERY_N == 0:

            # Grayscale + CLAHE on downsampled frame for speed
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if clahe:
                gray_full = apply_clahe_gray(gray_full, clahe)

            small = cv2.resize(gray_full, (0, 0),
                               fx=FRAME_SCALE, fy=FRAME_SCALE)

            # Detect faces in the small frame
            small_faces = detect_faces(small, cascade)

            # Scale coordinates back to full frame
            full_faces = [
                (int(x * inv), int(y * inv),
                 int(w * inv), int(h * inv))
                for (x, y, w, h) in small_faces
            ]

            new_labels   = []
            active_ids   = set(range(len(full_faces)))

            for fid, (fx, fy, fw, fh) in enumerate(full_faces):

                # Crop face from full-res gray frame and normalise size
                face_crop = gray_full[fy:fy + fh, fx:fx + fw]
                face_crop = cv2.resize(face_crop, FACE_SIZE)

                # LBPH prediction — returns (label_id, distance)
                # Lower distance = better match
                try:
                    pred_label, lbph_conf = recognizer.predict(face_crop)
                except cv2.error as e:
                    log.debug("LBPH predict error: %s", e)
                    new_labels.append(("Unknown", 0.0, COL_UNKNOWN, False))
                    continue

                # Convert LBPH distance to a 0–100% display score
                # (purely for the on-screen bar — not the decision threshold)
                display_pct = max(0.0, 100.0 - lbph_conf)

                if lbph_conf < LBPH_THRESHOLD:
                    # ── Match accepted ────────────────────────────────────
                    proposed  = label_map.get(pred_label, "Unknown")
                    confirmed = buf.update(fid, proposed)
                    pending   = buf.is_pending(fid)

                    if confirmed:
                        label = confirmed
                        color = COL_KNOWN
                        writer.record(label)
                        log.debug("CONFIRMED: %s  dist=%.1f", label, lbph_conf)
                    elif pending:
                        label = proposed   # Show tentative name while confirming
                        color = COL_PENDING
                    else:
                        label = proposed
                        color = COL_PENDING

                else:
                    # ── Match rejected (too low confidence) ───────────────
                    label = "Unknown"
                    color = COL_UNKNOWN
                    buf.update(fid, "Unknown")
                    log.debug("REJECTED: best=%s  dist=%.1f (threshold=%.1f)",
                              label_map.get(pred_label, "?"),
                              lbph_conf, LBPH_THRESHOLD)

                new_labels.append((label, display_pct, color,
                                   buf.is_pending(fid)))

            buf.clear_stale(active_ids)
            cached_faces  = full_faces
            cached_labels = new_labels

        # ── Draw cached annotations on every frame ────────────────────────
        for i, (fx, fy, fw, fh) in enumerate(cached_faces):
            if i < len(cached_labels):
                lbl, conf_pct, col, pend = cached_labels[i]
                draw_face_box(frame, fx, fy, fw, fh,
                              lbl, conf_pct, col, pend)

        draw_hud(frame, len(cached_faces), fps)
        cv2.imshow("Attendance System  –  LBPH Edition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("Q pressed – shutting down.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
#  SECTION 10 — POST-SESSION TERMINAL REPORT
# ═══════════════════════════════════════════════════════════════

def print_report(filepath: str) -> None:
    try:
        df = pd.read_csv(filepath)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        log.info("No attendance records found.")
        return

    if df.empty:
        log.info("Attendance file is empty.")
        return

    today    = datetime.now().strftime("%Y-%m-%d")
    today_df = df[df["Date"] == today]
    sep = "═" * 54

    print(f"\n{sep}\n  ATTENDANCE REPORT\n{sep}")
    if today_df.empty:
        print(f"  No entries for today ({today}).")
    else:
        print(f"  Date    : {today}")
        print(f"  Present : {len(today_df)}\n")
        for _, row in today_df.iterrows():
            print(f"    ✔  {row['Name']:<22}  {row['Time']}")

    print("\n  All-time totals:")
    totals = df.groupby("Name").size().reset_index(name="Days")
    for _, row in totals.iterrows():
        print(f"    {row['Name']:<22}  {row['Days']} day(s)")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 54)
    print("  Smart Attendance System  —  LBPH Edition (No dlib)")
    print("=" * 54)

    # 1. Verify opencv-contrib-python is installed
    try:
        _check_contrib()
    except ImportError as e:
        print(e)
        return

    # 2. Load cascade classifier
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        log.error("Could not load Haar cascade from: %s", CASCADE_PATH)
        return

    # 3. Build LBPH recognizer from dataset
    try:
        recognizer, label_map, _ = build_lbph_recognizer(DATASET_DIR, cascade)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return

    # 4. Set up attendance writer
    writer = AttendanceWriter(ATTENDANCE_FILE)

    # 5. Run recognition loop (webcam opens here automatically)
    try:
        recognize_faces(recognizer, label_map, cascade, writer)
    except IOError as e:
        log.error("%s", e)
        return

    # 6. Print session report
    print_report(ATTENDANCE_FILE)
    log.info("Session complete. Attendance saved to '%s'.", ATTENDANCE_FILE)


if __name__ == "__main__":
    main()