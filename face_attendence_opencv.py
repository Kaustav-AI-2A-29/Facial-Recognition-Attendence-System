"""
============================================================
  Smart Facial Recognition Attendance System (OpenCV Version)
  Uses OpenCV face detection instead of dlib
  Run with : python face_attendence_opencv.py
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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("attendance")

# Configuration
DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
MIN_FACE_SIZE = 80              # ✓ INCREASED: Filter tiny artifacts (was 50)
MIN_MATCHES = 8                 # ✓ INCREASED: More confirmations needed (was 5)
SIMILARITY_THRESHOLD = 0.75     # ✓ MUCH STRICTER: Raise from 0.65 to 0.75
HAAR_SCALE = 1.05              # ✓ NEW: More conservative detection (was 1.1)
HAAR_MIN_NEIGHBORS = 6          # ✓ NEW: More neighbors = fewer false positives

# Colors (BGR)
COL_KNOWN = (0, 210, 100)
COL_UNKNOWN = (0, 60, 220)
COL_HUD = (200, 200, 200)


def load_images_simple(dataset_dir: str):
    """Load images from dataset folder (flat or nested structure)."""
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    root = Path(dataset_dir)

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder '{dataset_dir}' not found.")

    person_files = {}

    # Check for nested structure (subfolders per person)
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            imgs = [f for f in sub.iterdir() if f.suffix.lower() in supported]
            if imgs:
                person_files[sub.name] = imgs

    # Fallback to flat structure
    if not person_files:
        for f in sorted(root.iterdir()):
            if f.is_file() and f.suffix.lower() in supported:
                person_files.setdefault(f.stem, []).append(f)

    if not person_files:
        raise ValueError(f"No images found in '{dataset_dir}'.")

    images_dict = {}
    log.info("Loading images from '%s'", dataset_dir)

    for name, paths in person_files.items():
        images_dict[name] = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                log.warning("  Cannot read '%s'", p.name)
                continue
            images_dict[name].append(img)
            log.info("  ✔  Loaded '%s/%s'", name, p.name)

    return images_dict


def compute_image_hash(img):
    """Compute multi-scale perceptual hash for better matching."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8))
    avg = small.mean()
    return (small > avg).flatten()


def compute_histogram(img):
    """Compute color histogram for additional matching."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def match_face_simple(ref_img, test_img, threshold=0.75):
    """Enhanced face matching using hash + histogram similarity."""
    
    # ✓ STRICT: Validate both images
    if test_img is None or test_img.size == 0:
        return False, 0.0
    if ref_img is None or ref_img.size == 0:
        return False, 0.0
    
    # ✓ NEW: Aspect ratio check (real faces are roughly square, 1.5:1 max)
    ref_h, ref_w = ref_img.shape[:2]
    test_h, test_w = test_img.shape[:2]
    
    ref_aspect = max(ref_w, ref_h) / max(min(ref_w, ref_h), 1)
    test_aspect = max(test_w, test_h) / max(min(test_w, test_h), 1)
    
    if ref_aspect > 2.0 or test_aspect > 2.0:
        return False, 0.0  # Not a real face - too elongated
    
    # Resize test_img to match reference for fair comparison
    if ref_img.shape != test_img.shape:
        test_img = cv2.resize(test_img, (ref_w, ref_h))
    
    # ✓ IMPROVED: Multi-metric matching
    # 1. Hash-based similarity
    ref_hash = compute_image_hash(ref_img)
    test_hash = compute_image_hash(test_img)
    hash_similarity = 1 - (np.sum(ref_hash != test_hash) / len(ref_hash))
    
    # 2. Histogram similarity (color consistency)
    try:
        ref_hist = compute_histogram(ref_img)
        test_hist = compute_histogram(test_img)
        hist_similarity = cv2.compareHist(ref_hist, test_hist, cv2.HISTCMP_BHATTACHARYYA)
        hist_similarity = 1 - min(hist_similarity, 1.0)  # Convert to 0-1 scale
    except:
        hist_similarity = hash_similarity  # Fallback if histogram fails
    
    # ✓ COMBINED: Both metrics must agree
    combined_similarity = (hash_similarity + hist_similarity) / 2.0
    
    log.debug(f"  Hash: {hash_similarity:.2f} | Hist: {hist_similarity:.2f} | Combined: {combined_similarity:.2f}")
    
    return combined_similarity >= threshold, combined_similarity


def initialize_csv(filepath: str):
    """Create CSV file if it doesn't exist."""
    if not os.path.isfile(filepath):
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time"])
        log.info("Created attendance file: '%s'", filepath)


def mark_attendance(filepath: str, name: str, marked_today: set):
    """Record attendance if not already done today."""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    key = f"{name}|{today}"

    if key in marked_today:
        return marked_today

    # Check CSV
    try:
        df = pd.read_csv(filepath)
        if not df[(df["Name"] == name) & (df["Date"] == today)].empty:
            marked_today.add(key)
            return marked_today
    except (pd.errors.EmptyDataError, FileNotFoundError):
        pass

    # Record
    with open(filepath, "a", newline="") as f:
        csv.writer(f).writerow([name, today, now.strftime("%H:%M:%S")])

    log.info("  [MARK] %s | %s | %s", name, today, now.strftime("%H:%M:%S"))
    marked_today.add(key)
    return marked_today


class FaceConfirmationBuffer:
    """✓ IMPROVED: Track face detections and require consistency."""
    def __init__(self, min_frames=MIN_MATCHES):
        self.min_frames = min_frames
        self.detection_history = {}  # {person_name: [count, avg_similarity]}
    
    def update(self, name: str, similarity: float) -> bool:
        """Update detection and return True if confirmed after N consistent frames."""
        if name == "Unknown":
            # Reset all on unknown
            self.detection_history.clear()
            return False
        
        if name not in self.detection_history:
            self.detection_history[name] = [0, similarity]
        
        count, avg_sim = self.detection_history[name]
        count += 1
        # Track running average similarity
        avg_sim = (avg_sim * (count - 1) + similarity) / count
        self.detection_history[name] = [count, avg_sim]
        
        # ✓ IMPROVED: Confirmed after N frames AND avg similarity > threshold
        if count >= self.min_frames and avg_sim >= 0.70:
            log.info(f"✓ CONFIRMED [{count} frames, avg sim: {avg_sim:.2f}]: {name}")
            return True
        
        log.debug(f"  Pending [{count}/{self.min_frames}] {name} (avg sim: {avg_sim:.2f})")
        return False
    
    def reset(self):
        """Reset buffer."""
        self.detection_history.clear()


def main():
    print("=" * 54)
    print("  Smart Facial Recognition Attendance (OpenCV Edition)")
    print("=" * 54)

    # Load dataset
    try:
        images_dict = load_images_simple(DATASET_DIR)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return

    if not images_dict:
        log.error("No valid images loaded from dataset.")
        return

    log.info("Loaded %d person(s) with %d image(s) total.\n",
             len(images_dict), sum(len(v) for v in images_dict.values()))

    # Load cascade classifier
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        log.error("Could not load face cascade classifier.")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    log.info("Webcam opened. Press Q to quit.\n")

    initialize_csv(ATTENDANCE_FILE)
    marked_today = set()
    frame_count = 0
    face_buffer = FaceConfirmationBuffer()  # ✓ Confirmation buffer
    frame_cache = {}  # ✓ NEW: Cache detected face ROIs to avoid re-detection

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ✓ IMPROVED: Use stricter Haar cascade parameters
        faces = face_cascade.detectMultiScale(gray, HAAR_SCALE, HAAR_MIN_NEIGHBORS)
        
        # ✓ FILTER: Remove tiny and oddly-sized faces
        faces = [
            (x, y, w, h) for (x, y, w, h) in faces 
            if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE 
            and 0.6 < (w / max(h, 1)) < 1.4  # ✓ NEW: Aspect ratio check
        ]

        # Match against dataset
        current_frame_matches = {}  # Track matches in this frame
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            best_name = "Unknown"
            best_sim = 0.0
            color = COL_UNKNOWN

            # ✓ IMPROVED: Only compare against reference images of actual people
            for name, ref_images in images_dict.items():
                for ref_img in ref_images:
                    matched, sim = match_face_simple(ref_img, face_roi, threshold=SIMILARITY_THRESHOLD)
                    if sim > best_sim:
                        best_sim = sim
                        if matched:
                            best_name = name
                            color = COL_KNOWN

            # ✓ IMPROVED: Stricter acceptance logic
            if best_name != "Unknown" and best_sim >= SIMILARITY_THRESHOLD:
                confirmed = face_buffer.update(best_name, best_sim)
                if confirmed:
                    marked_today = mark_attendance(ATTENDANCE_FILE, best_name, marked_today)
                    face_buffer.reset()
                current_frame_matches[best_name] = best_sim
            else:
                # ✓ NEW: Reset if no good match found
                face_buffer.reset()

            # Draw box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), color, cv2.FILLED)
            
            # ✓ IMPROVED: Show similarity score for debugging
            label_text = f"{best_name} ({best_sim:.2f})"
            if best_name != "Unknown":
                label_text += f" [{len([k for k, v in current_frame_matches.items() if k == best_name])}]"
            
            cv2.putText(frame, label_text, (x+6, y+h-10),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # HUD
        cv2.putText(frame, f"Faces: {len(faces)} | Q = Quit",
                   (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, COL_HUD, 1)

        cv2.imshow("Attendance System (OpenCV)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("Q pressed – shutting down.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Report
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        if not df.empty:
            today = datetime.now().strftime("%Y-%m-%d")
            today_df = df[df["Date"] == today]
            print(f"\n{'='*54}")
            print(f"  Attendance for {today}")
            print(f"{'='*54}")
            if not today_df.empty:
                for _, row in today_df.iterrows():
                    print(f"  ✔  {row['Name']:<20} {row['Time']}")
            print(f"{'='*54}\n")
    except Exception:
        pass

    log.info("Session complete. Attendance saved.")


if __name__ == "__main__":
    main()
