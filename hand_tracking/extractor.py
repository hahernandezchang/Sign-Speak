import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import torch
import json

# ─────────────────────────────────────────────
#  MODERN MEDIAPIPE IMPORTS
# ─────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

# ─────────────────────────────────────────────
#  AUTO-DOWNLOAD MEDIAPIPE MODEL
# ─────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded.")

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
WRIST      = 0
MIDDLE_MCP = 9
THUMB_TIP  = 4
INDEX_TIP  = 8

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ─────────────────────────────────────────────
#  COORDINATE HELPERS
# ─────────────────────────────────────────────
def extract_flat_coords(hand_landmarks) -> list[float]:
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords  # 63 floats


def normalize_coords(flat_coords: list[float]) -> list[float]:
    coords = np.array(flat_coords, dtype=np.float32).reshape(21, 3)
    origin = coords[WRIST].copy()
    coords -= origin
    scale = np.linalg.norm(coords[MIDDLE_MCP])
    if scale < 1e-6:
        return [0.0] * 63
    coords /= scale
    return coords.flatten().tolist()  # 63 floats


def compute_velocity(current: list[float], previous: list[float]) -> list[float]:
    """Frame-to-frame delta — encodes motion for J and Z."""
    return [c - p for c, p in zip(current, previous)]  # 63 floats


# ─────────────────────────────────────────────
#  ASL NEURAL NET  (must match train.py)
# ─────────────────────────────────────────────
class ASLNet(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [
                torch.nn.Linear(prev, h),
                torch.nn.BatchNorm1d(h),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            prev = h
        layers.append(torch.nn.Linear(prev, num_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
#  LOAD TRAINED CLASSIFIER
# ─────────────────────────────────────────────
_model     = None
_label_map = None

if os.path.exists("asl_classifier.pt") and os.path.exists("asl_label_map.json"):
    _ckpt      = torch.load("asl_classifier.pt", map_location="cpu")
    _label_map = json.load(open("asl_label_map.json"))
    _model     = ASLNet(
        _ckpt["input_size"],
        _ckpt["hidden_sizes"],
        _ckpt["num_classes"],
        _ckpt["dropout"],
    )
    _model.load_state_dict(_ckpt["model_state"])
    _model.eval()
    print(f"Classifier loaded — {_ckpt['num_classes']} classes: {list(_label_map.values())}")
else:
    print("No classifier found — running in landmark-only mode.")
    print("Run python train.py after collecting data.")


def classify_asl(position: list[float], velocity: list[float]) -> str | None:
    """Takes the 63-float position AND 63-float velocity → returns predicted letter."""
    if _model is None:
        return None
    features = position + velocity                          # 126 floats
    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        idx = _model(x).argmax(1).item()
    return _label_map[str(idx)]


# ─────────────────────────────────────────────
#  DRAWING UTILITIES
# ─────────────────────────────────────────────
def draw_hand(frame, hand_landmarks):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2, cv2.LINE_AA)
    fingertips = {4, 8, 12, 16, 20}
    for idx, (x, y) in enumerate(pts):
        color  = (0, 80, 255) if idx in fingertips else (255, 255, 255)
        radius = 6 if idx in fingertips else 4
        cv2.circle(frame, (x, y), radius, color,  -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, (0, 0, 0), 1, cv2.LINE_AA)


def draw_status(frame, hand_landmarks, hand_index: int):
    t = hand_landmarks[THUMB_TIP]
    i = hand_landmarks[INDEX_TIP]
    dist     = ((t.x - i.x) ** 2 + (t.y - i.y) ** 2) ** 0.5
    is_pinch = dist < 0.05
    label    = "PINCH" if is_pinch else "OPEN"
    color    = (0, 80, 255) if is_pinch else (0, 220, 120)
    cv2.putText(frame, label, (30, 80 + hand_index * 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3, cv2.LINE_AA)


def draw_motion_meter(frame, velocity: list[float]):
    """Shows a motion magnitude bar — helpful visual for J/Z."""
    mag   = float(np.linalg.norm(velocity))
    bar_w = int(min(mag * 400, 200))
    color = (0, 200, 255) if mag > 0.05 else (60, 60, 60)
    bx, by = frame.shape[1] - 220, frame.shape[0] - 30
    cv2.rectangle(frame, (bx, by), (bx + 200, by + 14), (40, 40, 40), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 14), color, -1)
    cv2.putText(frame, f"motion {mag:.3f}", (bx, by - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    print("Starting ASL extractor. Press Q to quit.")

    # per-hand velocity buffers keyed by hand index
    prev_norm = {}

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            result       = landmarker.detect_for_video(mp_image, timestamp_ms)

            active_hand_indices = set()

            if result.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                    active_hand_indices.add(hand_idx)

                    # coords
                    raw        = extract_flat_coords(hand_landmarks)
                    norm       = normalize_coords(raw)
                    last_norm  = prev_norm.get(hand_idx, [0.0] * 63)
                    velocity   = compute_velocity(norm, last_norm)
                    prev_norm[hand_idx] = norm

                    # draw
                    draw_hand(frame, hand_landmarks)
                    draw_status(frame, hand_landmarks, hand_idx)
                    draw_motion_meter(frame, velocity)

                    # classify
                    prediction = classify_asl(norm, velocity)
                    if prediction:
                        cv2.putText(
                            frame,
                            f"ASL: {prediction.upper()}",
                            (30, 200 + hand_idx * 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                            (0, 255, 255), 4, cv2.LINE_AA,
                        )

            # clear velocity buffers for hands that disappeared
            for idx in list(prev_norm.keys()):
                if idx not in active_hand_indices:
                    del prev_norm[idx]

            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f"{fps:.0f} FPS", (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

            cv2.imshow("ASL Landmark Extractor", frame)
            if cv2.waitKey(1) & 0xFF == ord("0"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()