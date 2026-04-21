import cv2
import mediapipe as mp
import pickle
import numpy as np

ESC_KEY_ASCII_CODE = 27

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)

model_chord = pickle.load(open("model_chord.pickle", "rb"))["model"]
model_stroke = pickle.load(open("model_stroke.pickle", "rb"))["model"]

# keys must match class indices used during data collection
chord_labels = {0: "C", 1: "G", 2: "Am"}
stroke_labels = {0: "Down", 1: "Up", 2: "Mute"}

FINGER_COLORS = [
    (HandLandmarksConnections.HAND_PALM_CONNECTIONS,          (255, 255, 255)),  # white
    (HandLandmarksConnections.HAND_THUMB_CONNECTIONS,         (0, 128, 255)),    # orange
    (HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS,  (0, 255, 0)),      # green
    (HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS, (255, 0, 0)),      # blue
    (HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS,   (0, 0, 255)),      # red
    (HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS,  (255, 0, 255)),    # magenta
]

def draw_connections(frame, hand_landmarks):
    height, width, _ = frame.shape
    for connections, color in FINGER_COLORS:
        for conn in connections:
            start = hand_landmarks[conn.start]
            end = hand_landmarks[conn.end]
            sx, sy = int(start.x * width), int(start.y * height)
            ex, ey = int(end.x * width), int(end.y * height)
            cv2.line(frame, (sx, sy), (ex, ey), color, 2)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # flip before MediaPipe so handedness labels are swapped:
        # MediaPipe "Right" = user's left hand (chord)
        # MediaPipe "Left"  = user's right hand (stroke)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detected_hand_data = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        if detected_hand_data.hand_landmarks:
            h, w, _ = frame.shape
            for i, hand in enumerate(detected_hand_data.hand_landmarks):
                draw_connections(frame, hand)

                coords = [v for lm in hand for v in (lm.x, lm.y)]
                side = detected_hand_data.handedness[i][0].category_name

                if side == "Right":
                    label = chord_labels[int(model_chord.predict([coords])[0])]
                else:
                    label = stroke_labels[int(model_stroke.predict([coords])[0])]

                wrist = hand[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, label, (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Ghost Riff", frame)

        if cv2.waitKey(1) & 0xFF == ESC_KEY_ASCII_CODE:
            break

cap.release()
cv2.destroyAllWindows()
