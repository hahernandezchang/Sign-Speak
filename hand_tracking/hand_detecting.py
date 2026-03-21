import cv2
import mediapipe as mp
import urllib.request
import os

# --- MODERN IMPORTS (Flat Path) ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- AUTO-DOWNLOAD MODEL ---
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# --- SETUP DETECTOR ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO, # Optimized for webcam
    num_hands=2
)

# Use 'with' to ensure the detector closes properly
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        # Convert to MediaPipe's format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Get timestamp in milliseconds
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # 1. Draw points manually
                for lm in hand_landmarks:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # 2. Pinch Logic (Thumb=4, Index=8)
                t, i = hand_landmarks[4], hand_landmarks[8]
                dist = ((t.x - i.x)**2 + (t.y - i.y)**2)**0.5
                
                status = "PINCH!" if dist < 0.05 else "Open"
                cv2.putText(frame, status, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow('MediaPipe 2026 Fix', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
