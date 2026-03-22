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

Output:  asl_data.csv  (one row per sample, dynamic feature count + label)
         Features include upper-body pose, both hands, and head anchors.
         CSV stores position features and frame-to-frame velocity features.

NOTE: if you have an old asl_data.csv from a previous feature layout,
    delete it and re-collect — formats are not compatible.
"""

import cv2
import numpy as np
import csv
import os
import time
import argparse
from collections import defaultdict

from feature_utils import (
    compute_velocity,
    create_holistic_model,
    draw_overlays,
    extract_position_features,
    feature_size,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "hand_tracking"
    else SCRIPT_DIR
)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "asl_data.csv")
SAMPLE_RATE = 10      # max samples saved per second while key is held

FEATURE_SIZE = feature_size()

os.makedirs(DATA_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ASL features with MediaPipe Holistic.")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g., 0) or path to a video file.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional fixed label for all captured samples (useful for prerecorded word videos).",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=SAMPLE_RATE,
        help="Max samples captured per second.",
    )
    return parser.parse_args()


def resolve_source(source: str):
    return int(source) if source.isdigit() else source


# ── load existing data count so we can resume ────────────────────────────────
sample_counts = defaultdict(int)
existing_rows = 0

if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and len(header) != (FEATURE_SIZE * 2) + 1:
            print("⚠️  WARNING: existing CSV has wrong number of columns.")
            print(f"   Expected {(FEATURE_SIZE * 2)} feature columns (position + velocity).")
            print("   Delete asl_data.csv and re-collect all letters.")
        else:
            for row in reader:
                if row:
                    sample_counts[row[-1]] += 1
                    existing_rows += 1
            print(f"Resuming — loaded {existing_rows} existing samples.")


# ── main collection loop ──────────────────────────────────────────────────────
def main():
    args = parse_args()

    csv_file = open(OUTPUT_CSV, "a", newline="")
    writer   = csv.writer(csv_file)

    # write header only for brand-new files
    if existing_rows == 0:
        pos_header = [f"f{i}" for i in range(FEATURE_SIZE)]
        vel_header = [f"df{i}" for i in range(FEATURE_SIZE)]
        writer.writerow(pos_header + vel_header + ["label"])

    last_capture_time = 0
    capture_interval  = 1.0 / max(args.sample_rate, 1e-6)
    session_counts    = defaultdict(int)

    # Velocity buffer for frame-to-frame motion features.
    prev_pos = [0.0] * FEATURE_SIZE

    print(f"\nCollector ready — {(FEATURE_SIZE * 2)}-feature mode (position + velocity).")
    if args.label:
        print(f"Fixed-label mode active: {args.label}")
        print("Samples will save automatically when hands are visible.")
    else:
        print("Hold a sign -> hold the matching letter key -> samples save automatically.")
        print("For J and Z: perform the actual motion while holding the key.")
    print("Press Q to quit.\n")

    with create_holistic_model() as holistic:
        cap = cv2.VideoCapture(resolve_source(str(args.source)))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            # ── which key is held this frame ──
            key = cv2.waitKey(1) & 0xFF
            active_label = args.label
            if active_label is None:
                if ord("a") <= key <= ord("z"):
                    active_label = chr(key)
                if ord("0") <= key <= ord("9"):
                    active_label = chr(key)

            # ── process landmarks ──
            hand_found = bool(result.left_hand_landmarks or result.right_hand_landmarks)
            pos = extract_position_features(result)
            velocity = compute_velocity(pos, prev_pos)
            prev_pos = pos

            draw_overlays(frame, result)

            vel_mag = np.linalg.norm(velocity)
            vel_color = (0, 200, 255) if vel_mag > 0.05 else (80, 80, 80)
            cv2.putText(
                frame,
                f"motion: {vel_mag:.3f}",
                (frame.shape[1] - 200, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                vel_color,
                1,
                cv2.LINE_AA,
            )

            now = time.time()
            if active_label and hand_found and (now - last_capture_time) >= capture_interval:
                writer.writerow(pos + velocity + [active_label])
                csv_file.flush()
                last_capture_time = now
                session_counts[active_label] += 1
                sample_counts[active_label] += 1

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
            if key == ord("q"):
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