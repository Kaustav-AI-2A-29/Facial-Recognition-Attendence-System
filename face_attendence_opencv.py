"""
Smart Face Recognition Attendance System (Pure OpenCV)
======================================================
Simple, reliable implementation using only OpenCV (no dlib).
Works on Windows without compilation issues.

Run:
    python face_attendence_opencv.py
"""

import cv2
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
from pathlib import Path


# ──────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────

DATASET_DIR     = "dataset"
ATTENDANCE_FILE = "attendance.csv"
CASCADE_PATH    = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
WEBCAM_INDEX    = 0
FRAME_RESIZE    = 0.5


# ──────────────────────────────────────────────────────────
# Image Hashing for Face Matching (no dlib needed)
# ──────────────────────────────────────────────────────────

def compute_face_hash(img_bgr, face_rect):
    """Extract face region and compute perceptual hash."""
    x, y, w, h = face_rect
    face = img_bgr[y:y+h, x:x+w]
    
    # Resize to 8x8 and convert to grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8))
    
    # Compute hash
    avg = small.mean()
    return (small > avg).flatten()


def hash_distance(hash1, hash2):
    """Hamming distance between two hashes (0-1, lower=more similar)."""
    if len(hash1) == 0 or len(hash2) == 0:
        return 1.0
    return np.sum(hash1 != hash2) / len(hash1)


# ──────────────────────────────────────────────────────────
# Load Dataset
# ──────────────────────────────────────────────────────────

def load_known_faces(dataset_dir, cascade):
    """
    Load images from dataset/<PersonName>/ folders.
    Extract face rectangles and compute hashes.
    Return: {person_name: [img_bgr, face_rects, hashes]}
    """
    known_faces = {}
    supported = (".jpg", ".jpeg", ".png", ".bmp")
    root = Path(dataset_dir)

    if not root.exists():
        print(f"\n[ERROR] '{dataset_dir}' folder not found.\n")
        return {}

    print(f"\nLoading dataset from '{dataset_dir}' ...\n")

    for person_folder in sorted(root.iterdir()):
        if not person_folder.is_dir():
            continue

        name = person_folder.name
        faces_for_person = []

        for img_path in sorted(person_folder.iterdir()):
            if img_path.suffix.lower() not in supported:
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            face_rects = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            
            if len(face_rects) == 0:
                print(f"  [skip] no face in {img_path.name}")
                continue

            # Use first face found
            rect = face_rects[0]
            h = compute_face_hash(img_bgr, rect)
            faces_for_person.append((img_bgr, rect, h))

        if faces_for_person:
            known_faces[name] = faces_for_person
            print(f"  OK  {name:<18} {len(faces_for_person)} image(s)")
        else:
            print(f"  X   {name:<18} no usable images")

    print(f"\n  {len(known_faces)} person(s) loaded.\n")
    return known_faces


# ──────────────────────────────────────────────────────────
# CSV Attendance
# ──────────────────────────────────────────────────────────

def setup_csv(filepath):
    """Create CSV file if missing."""
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time"])


def mark_attendance(name, filepath, marked_set):
    """Record attendance once per session."""
    if name in marked_set:
        return

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    t = now.strftime("%H:%M:%S")

    # Check CSV
    try:
        df = pd.read_csv(filepath)
        if not df[(df["Name"] == name) & (df["Date"] == today)].empty:
            marked_set.add(name)
            return
    except (pd.errors.EmptyDataError, FileNotFoundError):
        pass

    marked_set.add(name)
    
    with open(filepath, "a", newline="") as f:
        csv.writer(f).writerow([name, today, t])

    print(f"  [MARKED] {name} — {today}  {t}")


# ──────────────────────────────────────────────────────────
# Webcam Loop
# ──────────────────────────────────────────────────────────

def run(known_faces, cascade):
    """
    Open webcam and detect/recognize faces.
    Use hash matching to identify people.
    """
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        for idx in range(1, 4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"[INFO] Using webcam index {idx}")
                break
        else:
            print("[ERROR] No webcam found.")
            return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"Webcam open. Press Q to quit.\n")

    setup_csv(ATTENDANCE_FILE)
    marked = set()
    detection_counts = {}  # Track detections: {name: count}
    confirmation_threshold = 15  # Require 15+ frames to confirm
    threshold = 0.35  # hash distance threshold (lower = stricter)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_rects = cascade.detectMultiScale(small_gray, 1.1, 4, minSize=(30, 30))

        detected_this_frame = set()  # Track which names appear in this frame

        for (x, y, bw, bh) in face_rects:
            # Compute hash of detected face
            face = small[y:y+bh, x:x+bw]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_small = cv2.resize(face_gray, (8, 8))
            avg = face_small.mean()
            test_hash = (face_small > avg).flatten()

            # Find best match
            best_name = "Unknown"
            best_dist = threshold
            color = (0, 0, 200)  # red

            for person_name, face_list in known_faces.items():
                for _, _, ref_hash in face_list:
                    dist = hash_distance(test_hash, ref_hash)
                    if dist < best_dist:
                        best_dist = dist
                        best_name = person_name
                        color = (0, 200, 80)  # green

            # Count detections (skip Unknown)
            if best_name != "Unknown":
                detected_this_frame.add(best_name)  # Mark as detected in this frame
                detection_counts[best_name] = detection_counts.get(best_name, 0) + 1
                
                # Mark attendance only if count reaches threshold AND not already marked
                if detection_counts[best_name] >= confirmation_threshold and best_name not in marked:
                    mark_attendance(best_name, ATTENDANCE_FILE, marked)
                    detection_counts[best_name] = 0  # Reset after marking

            # Scale back to full frame
            sc = 1.0 / FRAME_RESIZE
            x1, y1 = int(x * sc), int(y * sc)
            x2, y2 = int((x + bw) * sc), int((y + bh) * sc)

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y2 - 36), (x2, y2), color, cv2.FILLED)
            cv2.putText(frame, best_name, (x1 + 6, y2 - 10),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        # Clear counts for people not in current frame
        for name in list(detection_counts.keys()):
            if name not in detected_this_frame:
                detection_counts[name] = 0

        # HUD: Show current detection counts
        counts_str = ", ".join([f"{n}:{c}" for n, c in detection_counts.items() if c > 0])
        hud_text = f"Faces: {len(face_rects)}  Q=Quit"
        if counts_str:
            hud_text += f"  [{counts_str}]"
        cv2.putText(frame, hud_text,
                   (10, 28), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)

        cv2.imshow("Face Attendance (OpenCV)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────

def print_report():
    """Print today's attendance summary."""
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        today = datetime.now().strftime("%Y-%m-%d")
        td = df[df["Date"] == today]
        
        sep = "=" * 44
        print(f"\n{sep}")
        print(f"  Attendance for {today}  ({len(td)} person(s))")
        print(sep)
        
        for _, r in td.iterrows():
            print(f"  [OK] {r['Name']:<18}  {r['Time']}")
        
        print(sep + "\n")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 44)
    print("  Face Recognition Attendance")
    print("  (OpenCV Pure - No dlib)")
    print("=" * 44)

    # Load cascade
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print("[ERROR] Could not load face cascade classifier.")
        return

    # Load dataset
    known_faces = load_known_faces(DATASET_DIR, cascade)

    if not known_faces:
        print("[ERROR] No faces loaded from dataset.")
        return

    run(known_faces, cascade)
    print_report()


if __name__ == "__main__":
    main()