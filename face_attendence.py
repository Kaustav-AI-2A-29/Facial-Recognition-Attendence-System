"""
============================================================
  Smart Facial Recognition Attendance System
  Author   : AI/ML Engineer (Claude)
  Stack    : Python 3.x | OpenCV | face_recognition | pandas
  Run with : python face_attendance.py
============================================================
"""

import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURATION  (edit these to suit your setup)
# ─────────────────────────────────────────────
DATASET_DIR       = "dataset"          # Folder containing reference images
ATTENDANCE_FILE   = "attendance.csv"   # Output CSV file
TOLERANCE         = 0.50               # Lower = stricter match (0.4–0.6 works well)
FRAME_SCALE       = 0.25              # Resize factor for faster processing
PROCESS_EVERY_N   = 2                  # Only process every Nth frame (reduces CPU load)
DISPLAY_FONT      = cv2.FONT_HERSHEY_DUPLEX
BOX_COLOR_KNOWN   = (0, 200, 100)      # Green  – recognised face
BOX_COLOR_UNKNOWN = (0, 0, 220)        # Red    – unknown face


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 ─ LOAD IMAGES FROM DATASET FOLDER
# ─────────────────────────────────────────────────────────────────────────────

def load_images(dataset_dir: str) -> tuple[list, list]:
    """
    Scan *dataset_dir* and return parallel lists of
    (person_names, PIL/numpy images).

    Rules
    -----
    - Filename (without extension) becomes the person's name.
    - Supported formats: .jpg  .jpeg  .png  .bmp
    - Images that fail to load or contain no detectable face are skipped.
    """
    supported = (".jpg", ".jpeg", ".png", ".bmp")
    images, names = [], []

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"[ERROR] Dataset folder '{dataset_dir}' not found.\n"
            f"        Create it and add at least one face image."
        )

    entries = [f for f in os.listdir(dataset_dir)
               if f.lower().endswith(supported)]

    if not entries:
        raise ValueError(
            f"[ERROR] No valid images found in '{dataset_dir}'.\n"
            f"        Supported formats: {supported}"
        )

    print(f"\n[INFO] Loading images from '{dataset_dir}' …")
    for filename in sorted(entries):
        path = os.path.join(dataset_dir, filename)
        img  = cv2.imread(path)

        if img is None:
            print(f"  [WARN] Could not read '{filename}' – skipping.")
            continue

        # Convert BGR (OpenCV default) → RGB (face_recognition requirement)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)

        # Strip extension to get the person's name  e.g. "Alice.jpg" → "Alice"
        name = os.path.splitext(filename)[0]
        names.append(name)
        print(f"  ✔  Loaded: {filename}  →  '{name}'")

    print(f"[INFO] {len(images)} image(s) loaded successfully.\n")
    return images, names


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 ─ ENCODE FACES FROM REFERENCE IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def encode_faces(images: list, names: list) -> tuple[list, list]:
    """
    Generate 128-dimensional face encodings for every loaded image.

    Returns parallel lists (known_encodings, known_names) where images
    that contain no detectable face are silently dropped.
    """
    known_encodings, known_names = [], []

    print("[INFO] Encoding faces …")
    for img_rgb, name in zip(images, names):
        encodings = face_recognition.face_encodings(img_rgb)

        if not encodings:
            print(f"  [WARN] No face detected in image for '{name}' – skipping.")
            continue

        # Use only the first face if multiple are present in a reference image
        known_encodings.append(encodings[0])
        known_names.append(name)
        print(f"  ✔  Encoded: '{name}'")

    if not known_encodings:
        raise ValueError(
            "[ERROR] No faces could be encoded from the dataset.\n"
            "        Ensure images contain clear, front-facing faces."
        )

    print(f"[INFO] {len(known_encodings)} face(s) encoded and ready.\n")
    return known_encodings, known_names


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 ─ ATTENDANCE CSV HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def initialise_csv(filepath: str) -> None:
    """
    Create the attendance CSV with a header row if it does not yet exist.
    """
    if not os.path.isfile(filepath):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])
        print(f"[INFO] Created attendance file: '{filepath}'")


def already_marked(filepath: str, name: str, today: str) -> bool:
    """
    Return True if *name* already has an entry for *today* in the CSV.
    Uses pandas for a clean, readable lookup.
    """
    try:
        df = pd.read_csv(filepath)
        return not df[(df["Name"] == name) & (df["Date"] == today)].empty
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return False


def mark_attendance(filepath: str, name: str,
                    marked_today: set) -> set:
    """
    Append a new attendance row for *name* if they haven't been marked yet
    during this session AND haven't been recorded in the CSV for today.

    Parameters
    ----------
    filepath     : path to the CSV file
    name         : recognised person's name
    marked_today : in-memory set of names already marked this session

    Returns the (possibly updated) *marked_today* set.
    """
    if name in marked_today:
        return marked_today  # Already handled this session – skip

    now   = datetime.now()
    today = now.strftime("%Y-%m-%d")
    time  = now.strftime("%H:%M:%S")

    if already_marked(filepath, name, today):
        print(f"  [SKIP] '{name}' already marked for {today}.")
        marked_today.add(name)          # Add to in-memory set to avoid repeated CSV reads
        return marked_today

    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, time])

    print(f"  [MARK] Attendance recorded ─ {name} | {today} | {time}")
    marked_today.add(name)
    return marked_today


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 ─ DRAW ANNOTATIONS ON FRAME
# ─────────────────────────────────────────────────────────────────────────────

def draw_annotation(frame, top, right, bottom, left,
                    label: str, confidence: float, color: tuple) -> None:
    """
    Draw a bounding box + name label + confidence score on *frame* (in-place).
    All coordinates are for the FULL-SIZE frame (already un-scaled).
    """
    # Bounding box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Filled label background
    cv2.rectangle(frame, (left, bottom - 38), (right, bottom),
                  color, cv2.FILLED)

    # Name text
    cv2.putText(frame, label, (left + 6, bottom - 18),
                DISPLAY_FONT, 0.65, (255, 255, 255), 1)

    # Confidence score below name
    conf_text = f"{confidence * 100:.1f}%" if confidence > 0 else ""
    cv2.putText(frame, conf_text, (left + 6, bottom - 4),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (220, 220, 220), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 ─ MAIN RECOGNITION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def recognize_faces(known_encodings: list, known_names: list) -> None:
    """
    Open the webcam and run the recognition loop until 'q' is pressed.

    Performance tricks
    ------------------
    - Frames are scaled to FRAME_SCALE before face detection.
    - Face detection runs only every PROCESS_EVERY_N frames.
    - The last detected results are re-drawn on skipped frames.
    """
    # ── Open webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError(
            "[ERROR] Cannot open webcam (index 0).\n"
            "        Check that your webcam is connected and not in use."
        )
    print("[INFO] Webcam opened. Press  Q  to quit.\n")

    # ── State variables ───────────────────────────────────────────────────────
    initialise_csv(ATTENDANCE_FILE)
    marked_today   : set  = set()      # Names marked this session
    frame_count    : int  = 0
    face_locations : list = []         # Cached from last processed frame
    face_labels    : list = []         # (label, confidence, color) per face
    inv_scale      : float = 1.0 / FRAME_SCALE

    # ── Loop ─────────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame – retrying …")
            continue

        frame_count += 1
        process_this_frame = (frame_count % PROCESS_EVERY_N == 0)

        if process_this_frame:
            # 1. Downsample for faster processing
            small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

            # 2. Convert BGR → RGB (face_recognition works in RGB)
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # 3. Detect face bounding boxes in the small frame
            face_locations = face_recognition.face_locations(small_rgb)

            # 4. Encode detected faces
            face_encodings = face_recognition.face_encodings(
                small_rgb, face_locations)

            face_labels = []   # Reset labels for this frame

            for enc in face_encodings:
                # Compare against every known encoding
                matches   = face_recognition.compare_faces(
                                known_encodings, enc, tolerance=TOLERANCE)
                distances = face_recognition.face_distance(known_encodings, enc)

                label      = "Unknown"
                confidence = 0.0
                color      = BOX_COLOR_UNKNOWN

                if len(distances) > 0:
                    best_idx  = int(np.argmin(distances))
                    best_dist = distances[best_idx]

                    # Convert distance to a 0-1 confidence score
                    confidence = max(0.0, 1.0 - best_dist)

                    if matches[best_idx]:
                        label = known_names[best_idx]
                        color = BOX_COLOR_KNOWN

                        # Record attendance (handles duplicates internally)
                        marked_today = mark_attendance(
                            ATTENDANCE_FILE, label, marked_today)

                face_labels.append((label, confidence, color))

        # ── Draw annotations (every frame, using cached data) ─────────────
        for (top, right, bottom, left), (label, conf, color) in zip(
                face_locations, face_labels):

            # Scale coordinates back to full-frame size
            top    = int(top    * inv_scale)
            right  = int(right  * inv_scale)
            bottom = int(bottom * inv_scale)
            left   = int(left   * inv_scale)

            draw_annotation(frame, top, right, bottom, left, label, conf, color)

        # ── HUD ─ frame counter + keybinding hint ─────────────────────────
        cv2.putText(frame,
                    f"Faces: {len(face_locations)}  |  Q = Quit",
                    (10, 28), cv2.FONT_HERSHEY_PLAIN, 1.4,
                    (200, 200, 200), 1)

        # ── Show window ───────────────────────────────────────────────────
        cv2.imshow("Smart Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] 'Q' pressed – shutting down …")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 ─ ATTENDANCE REPORT  (pandas summary)
# ─────────────────────────────────────────────────────────────────────────────

def print_attendance_report(filepath: str) -> None:
    """
    Load the CSV and print a quick summary to the terminal.
    Groups entries by date and lists who was marked present.
    """
    try:
        df = pd.read_csv(filepath)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("[INFO] No attendance records found yet.")
        return

    if df.empty:
        print("[INFO] Attendance file is empty.")
        return

    print("\n" + "═" * 52)
    print("          ATTENDANCE REPORT")
    print("═" * 52)

    today = datetime.now().strftime("%Y-%m-%d")
    today_df = df[df["Date"] == today]

    if today_df.empty:
        print(f"  No entries recorded for today ({today}).")
    else:
        print(f"  Date   : {today}")
        print(f"  Present: {len(today_df)} person(s)")
        print()
        for _, row in today_df.iterrows():
            print(f"    ✔  {row['Name']:<20} {row['Time']}")

    print("\n  All-time totals by person:")
    totals = df.groupby("Name").size().reset_index(name="Days Present")
    for _, row in totals.iterrows():
        print(f"    {row['Name']:<20} {row['Days Present']} day(s)")

    print("═" * 52 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 52)
    print("   Smart Facial Recognition Attendance System")
    print("=" * 52)

    try:
        # Step 1 – Load reference images
        images, names = load_images(DATASET_DIR)

        # Step 2 – Encode the loaded faces
        known_encodings, known_names = encode_faces(images, names)

        # Step 3 – Run real-time recognition loop
        recognize_faces(known_encodings, known_names)

    except (FileNotFoundError, ValueError, IOError) as exc:
        print(f"\n{exc}")
        return

    # Step 4 – Print a summary report after the session ends
    print_attendance_report(ATTENDANCE_FILE)
    print("[INFO] Session complete. Attendance saved to:", ATTENDANCE_FILE)


if __name__ == "__main__":
    main()