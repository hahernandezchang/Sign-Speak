"""
collector.py — ASL Training Data Collector
───────────────────────────────────────────
HOW TO USE:
  1. Run:  python collector.py
  2. Hold up a hand sign, press the corresponding letter key (a–z)
     while holding the pose to capture samples continuously.
  3. For J and Z — actually perform the motion while holding the key.
  4. Release the key to stop capturing that letter.
  5. Press Q to quit and save.

Output:  asl_data.csv  (one row per sample, 126 floats + label)
         Cols  0–62  = normalized position  (x0,y0,z0 ... x20,y20,z20)
         Cols 63–125 = velocity / delta     (current frame minus previous)

NOTE: if you have an old asl_data.csv with only 63 columns you must
      delete it and re-collect — the formats are not compatible.
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import urllib.request
import time
from collections import defaultdict

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

MODEL_PATH  = "hand_landmarker.task"
OUTPUT_CSV  = "asl_data.csv"
SAMPLE_RATE = 10      # max samples saved per second while key is held
WRIST       = 0
MIDDLE_MCP  = 9

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    urllib.request.urlretrieve(url, MODEL_PATH)


# ── coord helpers ─────────────────────────────────────────────────────────────
def extract_flat_coords(hand_landmarks):
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords  # 63 floats

def normalize_coords(flat_coords):
    coords = np.array(flat_coords, dtype=np.float32).reshape(21, 3)
    origin = coords[WRIST].copy()
    coords -= origin
    scale = np.linalg.norm(coords[MIDDLE_MCP])
    if scale < 1e-6:
        return [0.0] * 63
    coords /= scale
    return coords.flatten().tolist()  # 63 floats

def compute_velocity(current: list[float], previous: list[float]) -> list[float]:
    """
    Subtracts previous normalized frame from current.
    Both are 63-float arrays so the result is also 63 floats.
    For static letters this will be near-zero.
    For J/Z the moving frames will have large nonzero deltas —
    that's exactly what lets the classifier distinguish them.
    """
    return [c - p for c, p in zip(current, previous)]


# ── load existing data count so we can resume ────────────────────────────────
sample_counts = defaultdict(int)
existing_rows = 0

if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and len(header) != 127:   # 126 features + 1 label
            print("⚠️  WARNING: existing CSV has wrong number of columns.")
            print("   Old format had 63 features; new format needs 126.")
            print("   Delete asl_data.csv and re-collect all letters.")
        else:
            for row in reader:
                if row:
                    sample_counts[row[-1]] += 1
                    existing_rows += 1
            print(f"Resuming — loaded {existing_rows} existing samples.")


# ── main collection loop ──────────────────────────────────────────────────────
def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
    )

    csv_file = open(OUTPUT_CSV, "a", newline="")
    writer   = csv.writer(csv_file)

    # write header only for brand-new files
    if existing_rows == 0:
        pos_header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        vel_header = [f"d{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        writer.writerow(pos_header + vel_header + ["label"])

    last_capture_time = 0
    capture_interval  = 1.0 / SAMPLE_RATE
    session_counts    = defaultdict(int)

    # velocity buffer — holds the previous frame's normalized coords
    prev_norm = [0.0] * 63   # starts at zero; first frame velocity will be ~0

    print("\nCollector ready — 126-feature mode (position + velocity).")
    print("Hold a sign → hold the matching letter key → samples save automatically.")
    print("For J and Z: perform the actual motion while holding the key.")
    print("Press Q to quit.\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts     = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            result = landmarker.detect_for_video(mp_img, ts)

            # ── which key is held this frame ──
            key = cv2.waitKey(1) & 0xFF
            active_label = None
            if ord("a") <= key <= ord("z"):
                active_label = chr(key)
            if ord("0") <= key <= ord("9"):
                active_label = chr(key)

            # ── process landmarks ──
            hand_found = False
            if result.hand_landmarks:
                lms        = result.hand_landmarks[0]
                hand_found = True
                raw        = extract_flat_coords(lms)
                norm       = normalize_coords(raw)
                velocity   = compute_velocity(norm, prev_norm)
                prev_norm  = norm   # slide the window forward

                # draw skeleton
                h, w = frame.shape[:2]
                pts  = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17),
                ]
                for a, b in connections:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2, cv2.LINE_AA)
                for idx, (x, y) in enumerate(pts):
                    r = 6 if idx in {4,8,12,16,20} else 4
                    cv2.circle(frame, (x, y), r, (255,255,255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), r, (0,0,0), 1, cv2.LINE_AA)

                # velocity magnitude indicator (useful for J/Z capture)
                vel_mag = np.linalg.norm(velocity)
                vel_color = (0, 200, 255) if vel_mag > 0.05 else (80, 80, 80)
                cv2.putText(frame, f"motion: {vel_mag:.3f}",
                            (frame.shape[1] - 200, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, vel_color, 1, cv2.LINE_AA)

                # ── save sample ──
                now = time.time()
                if active_label and (now - last_capture_time) >= capture_interval:
                    writer.writerow(norm + velocity + [active_label])
                    csv_file.flush()
                    last_capture_time           = now
                    session_counts[active_label] += 1
                    sample_counts[active_label]  += 1
            else:
                # no hand detected — reset velocity buffer so stale delta
                # doesn't bleed into the next detected hand
                prev_norm = [0.0] * 63

            # ── HUD ──────────────────────────────────────────────────────────
            if active_label:
                color     = (0, 80, 255) if hand_found else (60, 60, 200)
                indicator = (f"SAVING: '{active_label.upper()}'"
                             if hand_found else f"NO HAND — '{active_label.upper()}'")
                cv2.putText(frame, indicator, (20, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Hold a letter key to capture",
                            (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (160, 160, 160), 1, cv2.LINE_AA)

            y = 100
            cv2.putText(frame, "Samples this session:", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            y += 22
            for letter in sorted(session_counts):
                total = sample_counts[letter]
                sess  = session_counts[letter]
                bar   = "█" * min(sess // 5, 20)
                text  = f"  {letter.upper()}: {total:4d} total  (+{sess}) {bar}"
                cv2.putText(frame, text, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 220, 160), 1)
                y += 18

            total_all = sum(sample_counts.values())
            cv2.putText(frame, f"Total saved: {total_all}",
                        (20, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)

            cv2.imshow("ASL Data Collector", frame)
            if key == ord("0"):
                break

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\n── Collection complete ──")
    print(f"Saved to: {OUTPUT_CSV}")
    for letter in sorted(sample_counts):
        print(f"  {letter.upper()}: {sample_counts[letter]} samples")
    print(f"  TOTAL: {sum(sample_counts.values())}")


if __name__ == "__main__":
    main()