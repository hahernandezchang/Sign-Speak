"""
Shared MediaPipe Holistic feature extraction for ASL training/inference.

Feature vector layout (position features):
- Upper-body pose anchors (x, y, z, visibility)
- Left hand 21 landmarks (x, y, z)
- Right hand 21 landmarks (x, y, z)
- Head/chin anchors from face mesh (x, y, z)
"""

from __future__ import annotations

from typing import Iterable
import warnings

import cv2
import mediapipe as mp
import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Focus on head + upper-body articulation for sign recognition.
POSE_INDICES = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
FACE_INDICES = [1, 10, 152, 234, 454]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def create_holistic_model(
    min_detection_confidence: float = 0.6,
    min_tracking_confidence: float = 0.6,
) -> mp_holistic.Holistic:
    return mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def _safe_get(landmarks, idx: int):
    if landmarks is None:
        return None
    if idx < 0 or idx >= len(landmarks):
        return None
    return landmarks[idx]


def _origin_and_scale(results) -> tuple[float, float, float]:
    pose = getattr(results, "pose_landmarks", None)
    if pose is None:
        return 0.5, 0.5, 0.25

    landmarks = pose.landmark
    left = _safe_get(landmarks, 11)
    right = _safe_get(landmarks, 12)
    if left is None or right is None:
        return 0.5, 0.5, 0.25

    cx = (left.x + right.x) * 0.5
    cy = (left.y + right.y) * 0.5
    shoulder_dist = float(np.linalg.norm([left.x - right.x, left.y - right.y]))
    scale = max(shoulder_dist, 1e-3)
    return cx, cy, scale


def _norm_xyz(x: float, y: float, z: float, cx: float, cy: float, scale: float) -> tuple[float, float, float]:
    return (x - cx) / scale, (y - cy) / scale, z / scale


def _pose_features(results, cx: float, cy: float, scale: float) -> list[float]:
    out: list[float] = []
    pose = getattr(results, "pose_landmarks", None)
    landmarks = pose.landmark if pose is not None else None

    for idx in POSE_INDICES:
        lm = _safe_get(landmarks, idx)
        if lm is None:
            out.extend([0.0, 0.0, 0.0, 0.0])
            continue
        nx, ny, nz = _norm_xyz(lm.x, lm.y, lm.z, cx, cy, scale)
        out.extend([nx, ny, nz, float(lm.visibility)])
    return out


def _hand_features(hand_landmarks, cx: float, cy: float, scale: float) -> list[float]:
    out: list[float] = []
    landmarks = hand_landmarks.landmark if hand_landmarks is not None else None
    for idx in range(21):
        lm = _safe_get(landmarks, idx)
        if lm is None:
            out.extend([0.0, 0.0, 0.0])
            continue
        nx, ny, nz = _norm_xyz(lm.x, lm.y, lm.z, cx, cy, scale)
        out.extend([nx, ny, nz])
    return out


def _face_features(results, cx: float, cy: float, scale: float) -> list[float]:
    out: list[float] = []
    face = getattr(results, "face_landmarks", None)
    landmarks = face.landmark if face is not None else None

    for idx in FACE_INDICES:
        lm = _safe_get(landmarks, idx)
        if lm is None:
            out.extend([0.0, 0.0, 0.0])
            continue
        nx, ny, nz = _norm_xyz(lm.x, lm.y, lm.z, cx, cy, scale)
        out.extend([nx, ny, nz])
    return out


def extract_position_features(results) -> list[float]:
    cx, cy, scale = _origin_and_scale(results)
    return (
        _pose_features(results, cx, cy, scale)
        + _hand_features(getattr(results, "left_hand_landmarks", None), cx, cy, scale)
        + _hand_features(getattr(results, "right_hand_landmarks", None), cx, cy, scale)
        + _face_features(results, cx, cy, scale)
    )


def feature_size() -> int:
    pose_dims = len(POSE_INDICES) * 4
    hand_dims = 21 * 3 * 2
    face_dims = len(FACE_INDICES) * 3
    return pose_dims + hand_dims + face_dims


def compute_velocity(current: Iterable[float], previous: Iterable[float]) -> list[float]:
    return [c - p for c, p in zip(current, previous)]


def extract_legacy_hand_position_features(results) -> list[float]:
    """
    Backward-compatible 63-float hand feature layout used by old checkpoints.
    Normalizes a single detected hand relative to wrist and middle MCP distance.
    """
    hand = getattr(results, "right_hand_landmarks", None)
    if hand is None:
        hand = getattr(results, "left_hand_landmarks", None)
    if hand is None:
        return [0.0] * 63

    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
    origin = coords[0].copy()
    coords -= origin
    scale = float(np.linalg.norm(coords[9]))
    if scale < 1e-6:
        return [0.0] * 63
    coords /= scale
    return coords.flatten().tolist()


def draw_overlays(frame, results) -> None:
    pose = getattr(results, "pose_landmarks", None)
    if pose is not None:
        h, w = frame.shape[:2]
        lms = pose.landmark
        for a, b in POSE_CONNECTIONS:
            la = _safe_get(lms, a)
            lb = _safe_get(lms, b)
            if la is None or lb is None:
                continue
            p1 = (int(la.x * w), int(la.y * h))
            p2 = (int(lb.x * w), int(lb.y * h))
            cv2.line(frame, p1, p2, (70, 190, 255), 2, cv2.LINE_AA)

        for idx in POSE_INDICES:
            lm = _safe_get(lms, idx)
            if lm is None:
                continue
            p = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)

    if getattr(results, "left_hand_landmarks", None) is not None:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 220, 120), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
        )

    if getattr(results, "right_hand_landmarks", None) is not None:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
        )

    face = getattr(results, "face_landmarks", None)
    if face is not None:
        h, w = frame.shape[:2]
        for idx in FACE_INDICES:
            lm = _safe_get(face.landmark, idx)
            if lm is None:
                continue
            p = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, p, 3, (255, 210, 70), -1, cv2.LINE_AA)