"""
============================================================
  Smart Facial Recognition Attendance System  v2.0
  Author   : AI/ML Engineer (Claude)
  Stack    : Python 3.x | OpenCV | face_recognition | pandas
  Run with : python face_attendance.py

  Improvements over v1.0
  ───────────────────────
  ① Averaged multi-image encodings per person (higher accuracy)
  ② CLAHE lighting normalisation (dim / back-lit rooms)
  ③ N-frame confirmation buffer (eliminates false positives)
  ④ Async CSV writer thread (zero I/O stall on main loop)
  ⑤ HOG + optional CNN model flag
  ⑥ Confidence bar drawn per face
  ⑦ Auto webcam fallback + detailed logging
============================================================
"""

import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
import csv
import threading
import queue
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
log = logging.getLogger("attendance")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit to suit your setup)
# ─────────────────────────────────────────────────────────────
DATASET_DIR      = "dataset"        # Sub-folders per person  OR  flat Name.jpg files
ATTENDANCE_FILE  = "attendance.csv"
TOLERANCE        = 0.48             # Lower = stricter (recommended: 0.45–0.52)
CONFIDENCE_MIN   = 0.52             # Min confidence (1-distance) to show a name
FRAME_SCALE      = 0.25             # Downsample factor before detection
PROCESS_EVERY_N  = 2                # Run detection on every Nth frame
CONFIRM_FRAMES   = 3                # Consecutive identical labels before accepting
FACE_MODEL       = "hog"            # "hog" (fast CPU) or "cnn" (accurate, needs GPU)
USE_CLAHE        = True             # Adaptive histogram equalisation for lighting
CLAHE_CLIP       = 2.0              # CLAHE clip limit (1.5–3.0)
CLAHE_TILE       = (8, 8)
WEBCAM_INDEX     = 0
WEBCAM_WIDTH     = 1280
WEBCAM_HEIGHT    = 720

# Display colours (BGR)
COL_KNOWN   = (0, 210, 100)         # Green  – confirmed known face
COL_UNKNOWN = (0, 60, 220)          # Red    – unknown face
COL_PENDING = (0, 190, 220)         # Amber  – confirmation in progress
COL_HUD     = (200, 200, 200)


# ═══════════════════════════════════════════════════════════════
#  SECTION 1  –  DATASET LOADER + AVERAGED ENCODING
# ═══════════════════════════════════════════════════════════════

def _read_rgb(path: str):
    """Read image file → RGB numpy array, or None on failure."""
    img = cv2.imread(path)
    if img is None:
        log.warning("Cannot read '%s' – skipped.", path)
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_and_encode_dataset(dataset_dir: str):
    """
    Load reference images and produce ONE averaged 128-D encoding per person.

    Supported layouts
    -----------------
    A) Flat    : dataset/Alice.jpg   dataset/Bob.jpg
    B) Nested  : dataset/Alice/img1.jpg   dataset/Alice/img2.jpg  ...

    Why averaging?
    --------------
    Encoding multiple photos of the same person under different lighting,
    poses, and expressions then averaging the 128-D vectors gives a centroid
    that is more representative than any single image.  Recognition then
    tolerates real-world variation far better while keeping impostors at bay.
    Recommended: 5–10 images per person for a 4-person dataset.
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    root = Path(dataset_dir)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder '{dataset_dir}' not found. "
            "Create it and add face images."
        )

    # Build  {name: [Path, …]}
    person_files: dict[str, list] = {}

    # Layout B – named sub-folders
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            imgs = [f for f in sub.iterdir() if f.suffix.lower() in supported]
            if imgs:
                person_files[sub.name] = imgs

    # Layout A – flat files (fallback)
    if not person_files:
        for f in sorted(root.iterdir()):
            if f.is_file() and f.suffix.lower() in supported:
                person_files.setdefault(f.stem, []).append(f)

    if not person_files:
        raise ValueError(
            f"No face images found in '{dataset_dir}'. "
            "Add .jpg/.png files or sub-folders named after each person."
        )

    known_encodings, known_names = [], []

    log.info("Encoding dataset from '%s' …", dataset_dir)
    for name, paths in person_files.items():
        enc_list = []
        for p in paths:
            img_rgb = _read_rgb(str(p))
            if img_rgb is None:
                continue
            encs = face_recognition.face_encodings(img_rgb)
            if not encs:
                log.warning("  No face in '%s' – skipped.", p.name)
                continue
            enc_list.append(encs[0])

        if not enc_list:
            log.warning("  No valid encodings for '%s' – person skipped.", name)
            continue

        avg = np.mean(enc_list, axis=0)   # Averaged robust descriptor
        known_encodings.append(avg)
        known_names.append(name)
        log.info("  ✔  %-20s  %d image(s) → averaged encoding", name, len(enc_list))

    if not known_encodings:
        raise ValueError(
            "No faces could be encoded from the dataset. "
            "Ensure images contain clear, front-facing faces."
        )

    log.info("Dataset ready: %d person(s).\n", len(known_encodings))
    return known_encodings, known_names


# ═══════════════════════════════════════════════════════════════
#  SECTION 2  –  LIGHTING NORMALISATION  (CLAHE)
# ═══════════════════════════════════════════════════════════════

def make_clahe():
    """
    Create CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Standard histogram eq amplifies noise in bright areas.
    CLAHE tiles the image and clips the contrast gain per tile,
    normalising dark/uneven faces without blowing out highlights.
    Applied to luminance only → skin tones stay realistic.
    """
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)


def apply_clahe(frame_bgr, clahe):
    """Apply CLAHE to luminance channel only (YCrCb space)."""
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# ═══════════════════════════════════════════════════════════════
#  SECTION 3  –  ASYNC CSV WRITER (background thread)
# ═══════════════════════════════════════════════════════════════

class AttendanceWriter:
    """
    Non-blocking attendance recorder.

    The 30-fps recognition loop cannot stall waiting for disk I/O.
    This class owns a daemon thread that drains a queue and writes
    CSV rows.  The main loop calls .record(name) and returns instantly.

    Duplicate guard: in-memory set (session) + pandas CSV lookup (restart).
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._queue   = queue.Queue()
        self._marked  = set()           # session-level guard
        self._lock    = threading.Lock()
        self._init_csv()
        threading.Thread(
            target=self._worker, daemon=True, name="csv-writer"
        ).start()
        log.info("CSV writer thread started → '%s'", filepath)

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

    def record(self, name: str):
        """Called from main thread.  Enqueues a write task and returns."""
        now   = datetime.now()
        today = now.strftime("%Y-%m-%d")
        key   = f"{name}|{today}"
        with self._lock:
            if key in self._marked:
                return
            self._marked.add(key)
        self._queue.put((name, today, now.strftime("%H:%M:%S")))

    def _worker(self):
        while True:
            name, today, t = self._queue.get()
            if not self._in_csv(name, today):
                with open(self.filepath, "a", newline="") as f:
                    csv.writer(f).writerow([name, today, t])
                log.info("  [MARK] %-20s %s  %s", name, today, t)
            self._queue.task_done()


# ═══════════════════════════════════════════════════════════════
#  SECTION 4  –  N-FRAME CONFIRMATION BUFFER
# ═══════════════════════════════════════════════════════════════

class ConfirmationBuffer:
    """
    Require CONFIRM_FRAMES consecutive identical labels before accepting.

    Why?
    ----
    A single motion-blurred or partially-occluded frame can push a face
    encoding just past the tolerance boundary of a wrong person.
    Requiring N frames to agree eliminates these one-shot false positives
    without adding meaningful latency (at 15 fps, N=3 adds ~200 ms).
    """

    def __init__(self):
        self._hist: dict[int, list] = {}

    def update(self, face_id: int, label: str):
        """Return confirmed label after CONFIRM_FRAMES identical hits, else None."""
        h = self._hist.setdefault(face_id, [])
        h.append(label)
        if len(h) > CONFIRM_FRAMES:
            h.pop(0)
        if len(h) == CONFIRM_FRAMES and len(set(h)) == 1:
            return label
        return None

    def is_pending(self, face_id: int) -> bool:
        h = self._hist.get(face_id, [])
        return 0 < len(h) < CONFIRM_FRAMES

    def clear_stale(self, active: set):
        for k in list(self._hist):
            if k not in active:
                del self._hist[k]


# ═══════════════════════════════════════════════════════════════
#  SECTION 5  –  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════

def draw_face_box(frame, top, right, bottom, left,
                  label, confidence, color, pending=False):
    """Draw bounding box + label strip + confidence bar."""
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    h = 44
    cv2.rectangle(frame, (left, bottom - h), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, label,
                (left + 6, bottom - h + 17),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    conf_txt = f"{confidence * 100:.1f}%"
    if pending:
        conf_txt += "  [confirming…]"
    cv2.putText(frame, conf_txt,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (220, 220, 220), 1, cv2.LINE_AA)

    if confidence > 0:
        bw = int((right - left) * min(confidence, 1.0))
        cv2.rectangle(frame,
                      (left, bottom - h - 5),
                      (left + bw, bottom - h - 1),
                      color, cv2.FILLED)


def draw_hud(frame, face_count, fps):
    cv2.putText(frame,
                f"Faces: {face_count}  |  {fps:.1f} fps  |  Q = Quit",
                (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.35, COL_HUD, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
#  SECTION 6  –  WEBCAM INITIALISATION
# ═══════════════════════════════════════════════════════════════

def open_webcam():
    """
    Open the webcam, falling back to adjacent indices if WEBCAM_INDEX fails.
    Sets preferred resolution and minimises buffer lag.
    Auto-start: the webcam is opened as soon as the program runs — no manual
    trigger needed.  The OS may show a permission dialog on first run.
    """
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        for idx in range(1, 4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                log.warning("Index %d failed; using index %d.", WEBCAM_INDEX, idx)
                break
        else:
            raise IOError(
                "No webcam detected. "
                "Ensure the camera is connected and not in use by another app."
            )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Reduces capture lag to ~1 frame
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Webcam opened at %dx%d. Press  Q  to quit.\n", w, h)
    return cap


# ═══════════════════════════════════════════════════════════════
#  SECTION 7  –  MAIN RECOGNITION LOOP
# ═══════════════════════════════════════════════════════════════

def recognize_faces(known_encodings, known_names, writer):
    """
    Real-time recognition and attendance loop.

    Per-frame pipeline
    ------------------
    1. Grab frame
    2. CLAHE lighting normalisation
    3. Downsample to FRAME_SCALE
    4. Detect face locations  (HOG or CNN model)
    5. Encode detected faces
    6. Compare → best match + confidence
    7. N-frame confirmation buffer
    8. Confirmed → AttendanceWriter.record()  (async, non-blocking)
    9. Draw annotations on full-size frame
    10. Display; loop
    """
    cap   = open_webcam()
    clahe = make_clahe() if USE_CLAHE else None
    buf   = ConfirmationBuffer()
    inv   = 1.0 / FRAME_SCALE

    frame_n    = 0
    fps_cnt    = 0
    fps        = 0.0
    fps_tick   = cv2.getTickCount()

    cached_locs   = []
    cached_labels = []    # List of (label, confidence, color, pending)

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame grab failed – retrying …")
            continue

        frame_n += 1
        fps_cnt += 1

        # FPS counter (updated every 30 frames)
        if fps_cnt >= 30:
            elapsed = (cv2.getTickCount() - fps_tick) / cv2.getTickFrequency()
            fps     = fps_cnt / max(elapsed, 1e-6)
            fps_cnt = 0
            fps_tick = cv2.getTickCount()

        # Only run detection every Nth frame
        if frame_n % PROCESS_EVERY_N == 0:
            src = apply_clahe(frame, clahe) if clahe else frame
            small = cv2.resize(src, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locs = face_recognition.face_locations(small_rgb, model=FACE_MODEL)
            encs = face_recognition.face_encodings(small_rgb, locs)

            new_labels   = []
            active_ids   = set(range(len(locs)))

            for fid, enc in enumerate(encs):
                distances = face_recognition.face_distance(known_encodings, enc)
                matches   = face_recognition.compare_faces(
                                known_encodings, enc, tolerance=TOLERANCE)

                label, conf, color = "Unknown", 0.0, COL_UNKNOWN

                if len(distances) > 0:
                    best = int(np.argmin(distances))
                    conf = max(0.0, 1.0 - distances[best])

                    if matches[best] and conf >= CONFIDENCE_MIN:
                        proposed  = known_names[best]
                        confirmed = buf.update(fid, proposed)
                        pending   = buf.is_pending(fid)

                        if confirmed:
                            label, color = confirmed, COL_KNOWN
                            writer.record(label)
                        elif pending:
                            label, color = proposed, COL_PENDING
                    else:
                        buf.update(fid, "Unknown")

                new_labels.append((label, conf, color, buf.is_pending(fid)))

            buf.clear_stale(active_ids)
            cached_locs   = locs
            cached_labels = new_labels

        # Draw on full-size frame
        for i, (top, right, bottom, left) in enumerate(cached_locs):
            t = int(top * inv); r = int(right * inv)
            b = int(bottom * inv); l = int(left * inv)
            if i < len(cached_labels):
                lbl, conf, col, pend = cached_labels[i]
                draw_face_box(frame, t, r, b, l, lbl, conf, col, pend)

        draw_hud(frame, len(cached_locs), fps)
        cv2.imshow("Smart Attendance System  v2.0", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("Q pressed – shutting down.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
#  SECTION 8  –  TERMINAL REPORT
# ═══════════════════════════════════════════════════════════════

def print_report(filepath: str):
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
            print(f"    ✔  {row['Name']:<22} {row['Time']}")

    print("\n  All-time totals:")
    for _, row in df.groupby("Name").size().reset_index(name="D").iterrows():
        print(f"    {row['Name']:<22} {row['D']} day(s)")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 54)
    print("  Smart Facial Recognition Attendance System  v2.0")
    print("=" * 54)

    try:
        known_enc, known_names = load_and_encode_dataset(DATASET_DIR)
        writer = AttendanceWriter(ATTENDANCE_FILE)
        recognize_faces(known_enc, known_names, writer)
    except (FileNotFoundError, ValueError, IOError) as exc:
        log.error("%s", exc)
        return

    print_report(ATTENDANCE_FILE)
    log.info("Session complete. Attendance saved to '%s'.", ATTENDANCE_FILE)


if __name__ == "__main__":
    main()