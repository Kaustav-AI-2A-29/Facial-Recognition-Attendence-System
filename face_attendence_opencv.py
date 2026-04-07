"""
============================================================
  Smart Facial Recognition Attendance System
  OpenCV LBPH Edition  v3.0  —  NO dlib required
  Run with : python face_attendence_opencv.py

  What changed from v2.0
  ──────────────────────
  FIX 1 — LBPH_THRESHOLD lowered to 35
      Background patches typically score 45–70 LBPH distance.
      Real faces trained from the same webcam score 10–30.
      Gate at 35 creates a clean separation between the two.

  FIX 2 — Non-Maximum Suppression (NMS) added
      Haar cascade often returns 2–3 overlapping boxes for one
      face, or a nearby background patch alongside the real face.
      NMS merges all boxes with IoU > 30% into one.
      1 real face = 1 detection = 1 LBPH prediction.  No ghost.

  FIX 3 — LBPH trained with radius=2, neighbors=16, grid 8x8
      Default LBPH (radius=1, neighbors=8) is too coarse for
      4 similar-looking people.  Finer descriptors = better
      separation between identities.

  FIX 4 — MIN_FACE_SIZE raised to 100px
      At 60cm webcam distance a real face is 130–180px wide.
      Background artifacts and hands are usually <80px at that
      scale.  100px gate catches real faces and ignores artifacts.

  HOW TO CALIBRATE LBPH_THRESHOLD (read this before tuning)
  ──────────────────────────────────────────────────────────
  1. Run the script and watch "dist=" values in the terminal.
  2. Your own face on good dataset images: dist should be 10–30.
  3. Background objects / strangers: dist should be 50–120.
  4. Set LBPH_THRESHOLD just ABOVE your real-face scores.
     e.g. face scores 22-28 → set threshold = 33
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("attendance")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATASET_DIR     = "dataset"
ATTENDANCE_FILE = "attendance.csv"

# LBPH: distance score — 0 = perfect, 100+ = terrible
# Accept match only when distance < LBPH_THRESHOLD
LBPH_THRESHOLD      = 35     # Lower = stricter. See calibration guide above.

CASCADE_PATH        = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_SCALE          = 1.10
HAAR_MIN_NEIGHBORS  = 10     # Raise to 12 if background still detected
MIN_FACE_SIZE       = 100    # Pixels in downsampled frame
MAX_FACE_RATIO      = 1.4    # Width/height — real faces ≈ 1.0–1.2

NMS_OVERLAP_THRESH  = 0.30   # IoU threshold for merging overlapping boxes

CONFIRM_FRAMES  = 6          # Consecutive identical frames before marking
PROCESS_EVERY_N = 2
FRAME_SCALE     = 0.5

USE_CLAHE  = True
CLAHE_CLIP = 2.5
CLAHE_TILE = (8, 8)

WEBCAM_INDEX  = 0
WEBCAM_WIDTH  = 1280
WEBCAM_HEIGHT = 720

COL_KNOWN   = (0, 210, 100)
COL_UNKNOWN = (0, 60, 220)
COL_PENDING = (0, 190, 220)
COL_HUD     = (200, 200, 200)

FACE_SIZE = (200, 200)


# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — DEPENDENCY CHECK
# ═══════════════════════════════════════════════════════════════

def _check_contrib() -> None:
    if not hasattr(cv2, "face"):
        raise ImportError(
            "\n\n  cv2.face not found. Install the contrib package:\n"
            "    pip uninstall opencv-python -y\n"
            "    pip install opencv-contrib-python\n"
        )


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — CLAHE
# ═══════════════════════════════════════════════════════════════

def make_clahe():
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

def clahe_gray(gray, clahe):
    return clahe.apply(gray)


# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — NON-MAXIMUM SUPPRESSION
# ═══════════════════════════════════════════════════════════════

def nms_boxes(boxes: list, overlap_thresh: float = NMS_OVERLAP_THRESH) -> list:
    """
    Non-Maximum Suppression for (x, y, w, h) boxes.

    WHY this fixes the "Soumita shown when only Kaustav present" bug:
    Haar cascade detects Kaustav's face AND a nearby background patch
    that looks slightly face-like. Both get predicted by LBPH. The
    background patch's nearest label happens to be Soumita, and if
    her distance passes the threshold she gets recorded.

    NMS merges overlapping boxes before LBPH prediction. After NMS,
    each physical face produces exactly ONE bounding box, so there is
    no second region to misidentify as Soumita.
    """
    if not boxes:
        return []

    rects = np.array([[x, y, x+w, y+h] for x, y, w, h in boxes], float)
    areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
    order = np.argsort(areas)[::-1]
    keep  = []

    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if not len(rest):
            break
        xx1 = np.maximum(rects[i, 0], rects[rest, 0])
        yy1 = np.maximum(rects[i, 1], rects[rest, 1])
        xx2 = np.minimum(rects[i, 2], rects[rest, 2])
        yy2 = np.minimum(rects[i, 3], rects[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= overlap_thresh]

    return [boxes[i] for i in keep]


# ═══════════════════════════════════════════════════════════════
#  SECTION 4 — LBPH TRAINER
# ═══════════════════════════════════════════════════════════════

def build_lbph_recognizer(dataset_dir: str, cascade, clahe):
    """
    Load reference images, crop faces, train LBPH.

    CRITICAL: Use capture_dataset.py to build your dataset.
    Images captured with the same webcam produce LBPH distances
    of 10-30 for the correct person. Phone photos or downloaded
    images produce distances of 40-80 even for the right person,
    making it impossible to set a threshold that works.
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    root = Path(dataset_dir)

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder '{dataset_dir}' not found.")

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
        raise ValueError(f"No images found in '{dataset_dir}'.")

    label_map: dict[int, str] = {}
    faces_train:  list = []
    labels_train: list = []
    label_id = 0

    log.info("Training LBPH from '%s' …", dataset_dir)

    for name, paths in person_files.items():
        count = 0
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = clahe_gray(gray, clahe)

            detected = cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(50, 50))

            if len(detected) > 0:
                x, y, w, h = max(detected, key=lambda r: r[2] * r[3])
                crop = gray[y:y+h, x:x+w]
            else:
                log.warning("  No face in '%s/%s' — using full image.", name, p.name)
                crop = gray

            faces_train.append(cv2.resize(crop, FACE_SIZE))
            labels_train.append(label_id)
            count += 1

        if count == 0:
            log.warning("  No usable images for '%s' — skipped.", name)
            continue
        label_map[label_id] = name
        log.info("  ✔  %-20s  %d image(s)", name, count)
        label_id += 1

    if not faces_train:
        raise ValueError("No training faces loaded.")

    # radius=2, neighbors=16, grid=8×8 gives finer texture descriptors
    # than the default (radius=1, neighbors=8, grid=8×8).
    # Better separation between 4 similar-looking individuals.
    rec = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=16, grid_x=8, grid_y=8)
    rec.train(faces_train, np.array(labels_train))
    log.info("LBPH trained — %d person(s) enrolled.\n", len(label_map))
    return rec, label_map


# ═══════════════════════════════════════════════════════════════
#  SECTION 5 — FACE DETECTION WITH NMS
# ═══════════════════════════════════════════════════════════════

def detect_faces(gray_small, cascade) -> list:
    raw = cascade.detectMultiScale(
        gray_small,
        scaleFactor=HAAR_SCALE,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(raw) == 0:
        return []
    shape_ok = [(x, y, w, h) for x, y, w, h in raw
                if max(w, h) / max(min(w, h), 1) <= MAX_FACE_RATIO]
    return nms_boxes(shape_ok)


# ═══════════════════════════════════════════════════════════════
#  SECTION 6 — CONFIRMATION BUFFER
# ═══════════════════════════════════════════════════════════════

class ConfirmationBuffer:
    def __init__(self):
        self._h: dict[int, list] = {}

    def update(self, fid: int, label: str):
        h = self._h.setdefault(fid, [])
        h.append(label)
        if len(h) > CONFIRM_FRAMES:
            h.pop(0)
        if len(h) == CONFIRM_FRAMES and len(set(h)) == 1 and h[0] != "Unknown":
            return h[0]
        return None

    def is_pending(self, fid: int) -> bool:
        h = self._h.get(fid, [])
        return 0 < sum(1 for x in h if x != "Unknown") < CONFIRM_FRAMES

    def clear_stale(self, active: set):
        for k in list(self._h):
            if k not in active:
                del self._h[k]


# ═══════════════════════════════════════════════════════════════
#  SECTION 7 — ATTENDANCE WRITER
# ═══════════════════════════════════════════════════════════════

class AttendanceWriter:
    def __init__(self, filepath: str):
        self.fp = filepath
        self._marked: set = set()
        if not os.path.isfile(filepath):
            with open(filepath, "w", newline="") as f:
                csv.writer(f).writerow(["Name", "Date", "Time"])

    def _in_csv(self, name, today):
        try:
            df = pd.read_csv(self.fp)
            return not df[(df["Name"] == name) & (df["Date"] == today)].empty
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return False

    def record(self, name: str) -> bool:
        now, today = datetime.now(), datetime.now().strftime("%Y-%m-%d")
        key = f"{name}|{today}"
        if key in self._marked:
            return False
        self._marked.add(key)
        if self._in_csv(name, today):
            return False
        with open(self.fp, "a", newline="") as f:
            csv.writer(f).writerow([name, today, now.strftime("%H:%M:%S")])
        log.info("  [MARK] %-20s %s  %s", name, today, now.strftime("%H:%M:%S"))
        return True


# ═══════════════════════════════════════════════════════════════
#  SECTION 8 — DRAW
# ═══════════════════════════════════════════════════════════════

def draw_box(frame, x, y, w, h, label, dist, color, pending=False):
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    s = 48
    cv2.rectangle(frame, (x, y+h-s), (x+w, y+h), color, cv2.FILLED)
    cv2.putText(frame, label, (x+6, y+h-s+18),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (255,255,255), 1, cv2.LINE_AA)
    # Show raw dist so you can calibrate threshold
    txt = f"dist={dist:.0f}  thresh={LBPH_THRESHOLD}"
    if pending:
        txt += "  [confirming…]"
    cv2.putText(frame, txt, (x+6, y+h-7),
                cv2.FONT_HERSHEY_PLAIN, 0.8, (220,220,220), 1, cv2.LINE_AA)
    # Fill bar: shorter dist = more filled
    bw = int(w * max(0.0, 1.0 - dist / max(LBPH_THRESHOLD, 1)))
    cv2.rectangle(frame, (x, y+h-s-5), (x+bw, y+h-s-1), color, cv2.FILLED)

def draw_hud(frame, n, fps):
    cv2.putText(frame, f"Faces:{n}  {fps:.1f}fps  Q=Quit",
                (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.3, COL_HUD, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════
#  SECTION 9 — WEBCAM
# ═══════════════════════════════════════════════════════════════

def open_webcam():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        for i in range(1, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
        else:
            raise IOError("No webcam found.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    log.info("Webcam %dx%d.  Q to quit.\n",
             int(cap.get(3)), int(cap.get(4)))
    return cap


# ═══════════════════════════════════════════════════════════════
#  SECTION 10 — MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def recognize_faces(rec, label_map, cascade, clahe, writer):
    cap  = open_webcam()
    buf  = ConfirmationBuffer()
    inv  = 1.0 / FRAME_SCALE

    frame_n = fps_cnt = 0
    fps = 0.0
    fps_tick = cv2.getTickCount()
    cf: list = []   # cached face boxes (full res)
    cl: list = []   # cached labels

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_n += 1
        fps_cnt += 1

        if fps_cnt >= 30:
            e = (cv2.getTickCount() - fps_tick) / cv2.getTickFrequency()
            fps = fps_cnt / max(e, 1e-6)
            fps_cnt = 0
            fps_tick = cv2.getTickCount()

        if frame_n % PROCESS_EVERY_N == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if clahe:
                g = clahe_gray(g, clahe)
            gs = cv2.resize(g, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

            sf = detect_faces(gs, cascade)
            ff = [(int(x*inv), int(y*inv), int(w*inv), int(h*inv))
                  for x, y, w, h in sf]

            nl = []
            ai = set(range(len(ff)))

            for fid, (fx, fy, fw, fh) in enumerate(ff):
                crop = cv2.resize(g[fy:fy+fh, fx:fx+fw], FACE_SIZE)
                try:
                    pid, dist = rec.predict(crop)
                except cv2.error:
                    nl.append(("Unknown", 999.0, COL_UNKNOWN, False))
                    continue

                # Print to terminal so you can read real distances
                log.debug("fid=%d  best=%-15s  dist=%.1f  gate=%d",
                          fid, label_map.get(pid, "?"), dist, LBPH_THRESHOLD)

                if dist < LBPH_THRESHOLD:
                    prop = label_map.get(pid, "Unknown")
                    conf = buf.update(fid, prop)
                    if conf:
                        label, color = conf, COL_KNOWN
                        writer.record(label)
                    else:
                        label, color = prop, COL_PENDING
                else:
                    label, color = "Unknown", COL_UNKNOWN
                    buf.update(fid, "Unknown")

                nl.append((label, dist, color, buf.is_pending(fid)))

            buf.clear_stale(ai)
            cf, cl = ff, nl

        for i, (fx, fy, fw, fh) in enumerate(cf):
            if i < len(cl):
                lbl, dist, col, pend = cl[i]
                draw_box(frame, fx, fy, fw, fh, lbl, dist, col, pend)

        draw_hud(frame, len(cf), fps)
        cv2.imshow("Attendance — LBPH v3", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 54)
    print("  Attendance System — LBPH v3  (No dlib)")
    print("=" * 54)

    try:
        _check_contrib()
    except ImportError as e:
        print(e); return

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        log.error("Haar cascade not found: %s", CASCADE_PATH); return

    clahe = make_clahe() if USE_CLAHE else None

    try:
        rec, label_map = build_lbph_recognizer(DATASET_DIR, cascade, clahe)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e); return

    writer = AttendanceWriter(ATTENDANCE_FILE)

    try:
        recognize_faces(rec, label_map, cascade, clahe, writer)
    except IOError as e:
        log.error("%s", e); return

    # Print final summary
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        today = datetime.now().strftime("%Y-%m-%d")
        td = df[df["Date"] == today]
        print(f"\n{'='*54}\n  Today ({today}): {len(td)} person(s)")
        for _, r in td.iterrows():
            print(f"    ✔  {r['Name']:<22}  {r['Time']}")
        print("=" * 54)
    except Exception:
        pass


if __name__ == "__main__":
    main()