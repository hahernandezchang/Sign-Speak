import os
import argparse

import cv2
import numpy as np

from feature_utils import (
    compute_velocity,
    create_holistic_model,
    draw_overlays,
    extract_legacy_hand_position_features,
    extract_position_features,
    feature_size,
)
from model_utils import load_classifier, predict_label

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "asl_classifier.pt")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "asl_label_map.json")
FEATURE_SIZE = feature_size()
EXPECTED_INPUT_SIZE = FEATURE_SIZE * 2
LEGACY_INPUT_SIZE = 126


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASL extractor with holistic features.")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g., 0) or path to a video file.",
    )
    parser.add_argument("--model-path", default=CLASSIFIER_PATH)
    parser.add_argument("--label-map-path", default=LABEL_MAP_PATH)
    return parser.parse_args()


def resolve_source(source: str):
    return int(source) if source.isdigit() else source


def draw_motion_meter(frame, velocity: list[float]):
    mag = float(np.linalg.norm(velocity))
    bar_w = int(min(mag * 120, 220))
    color = (0, 200, 255) if mag > 0.05 else (60, 60, 60)
    bx, by = frame.shape[1] - 240, frame.shape[0] - 30
    cv2.rectangle(frame, (bx, by), (bx + 220, by + 14), (40, 40, 40), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 14), color, -1)
    cv2.putText(
        frame,
        f"motion {mag:.3f}",
        (bx, by - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    classifier = None
    if os.path.exists(args.model_path) and os.path.exists(args.label_map_path):
        loaded = load_classifier(args.model_path, args.label_map_path)
        if loaded.input_size == LEGACY_INPUT_SIZE:
            classifier = loaded
            print("Classifier loaded in legacy 126-feature compatibility mode.")
            print(
                f"Classes: {len(classifier.label_map)} -> {list(classifier.label_map.values())}"
            )
        elif loaded.input_size != EXPECTED_INPUT_SIZE:
            print(
                "Classifier input mismatch. "
                f"Checkpoint expects {loaded.input_size}, runtime produces {EXPECTED_INPUT_SIZE}."
            )
            print("Retrain with current collector/train scripts, or provide a compatible checkpoint.")
        else:
            classifier = loaded
            print(
                f"Classifier loaded - {len(classifier.label_map)} classes: "
                f"{list(classifier.label_map.values())}"
            )
    else:
        print("No classifier found - running in landmark-only mode.")
        print("Run train.py with model-specific outputs first.")

    print("Starting ASL extractor (upper-body + hands + head). Press Q to quit.")
    print(f"Position feature size: {FEATURE_SIZE}; full model input size: {FEATURE_SIZE * 2}")

    prev_pos = [0.0] * FEATURE_SIZE
    prev_legacy_pos = [0.0] * 63

    with create_holistic_model() as holistic:
        cap = cv2.VideoCapture(resolve_source(str(args.source)))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            position = extract_position_features(results)
            velocity = compute_velocity(position, prev_pos)
            prev_pos = position

            legacy_position = extract_legacy_hand_position_features(results)
            legacy_velocity = compute_velocity(legacy_position, prev_legacy_pos)
            prev_legacy_pos = legacy_position

            draw_overlays(frame, results)
            draw_motion_meter(frame, velocity)

            prediction_result = None
            if classifier is not None:
                features = (
                    legacy_position + legacy_velocity
                    if classifier.input_size == LEGACY_INPUT_SIZE
                    else position + velocity
                )
                prediction_result = predict_label(classifier, features)
            if prediction_result:
                prediction, confidence = prediction_result
                cv2.putText(
                    frame,
                    f"ASL: {prediction}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"conf {confidence:.2f}",
                    (30, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "Tracking: shoulders, elbows, wrists, hands, head",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("ASL Holistic Extractor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
