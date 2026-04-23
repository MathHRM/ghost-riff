import cv2
import mediapipe as mp
import pickle
import time
import pygame
from labels import chord_labels, stroke_labels

ESC_KEY_ASCII_CODE = 27
MIN_DETECTION_CONFIDENCE = 0.6

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

pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()

chord_sounds = {}
for _idx, _info in chord_labels.items():
    _path = _info.get("audio")
    if _path is None:
        continue
    try:
        snd = pygame.mixer.Sound(_path)
        chord_sounds[_info["name"]] = {"path": _path, "duration": snd.get_length()}
    except Exception:
        pass

current_chord_name = None
last_stroke_name = None
last_play_time = None

FINGER_COLORS = [
    (HandLandmarksConnections.HAND_PALM_CONNECTIONS,          (255, 255, 255)),  # white
    (HandLandmarksConnections.HAND_THUMB_CONNECTIONS,         (0, 128, 255)),    # orange
    (HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS,  (0, 255, 0)),      # green
    (HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS, (255, 0, 0)),      # blue
    (HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS,   (0, 0, 255)),      # red
    (HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS,  (255, 0, 255)),    # magenta
]

def handle_stroke(stroke_name):
    global last_stroke_name, last_play_time

    if stroke_name == last_stroke_name:
        return

    last_stroke_name = stroke_name

    if stroke_name == "Mute":
        pygame.mixer.music.stop()
        return

    if stroke_name == "Idle" or current_chord_name not in chord_sounds:
        return

    if last_play_time is not None:
        elapsed = time.monotonic() - last_play_time
        if elapsed < chord_sounds[current_chord_name]["duration"] * 0.5:
            return

    try:
        pygame.mixer.music.load(chord_sounds[current_chord_name]["path"])
        pygame.mixer.music.play()
        last_play_time = time.monotonic()
    except Exception:
        pass

def draw_connections(frame, hand_landmarks):
    height, width, _ = frame.shape
    for connections, color in FINGER_COLORS:
        for conn in connections:
            start = hand_landmarks[conn.start]
            end = hand_landmarks[conn.end]
            sx, sy = int(start.x * width), int(start.y * height)
            ex, ey = int(end.x * width), int(end.y * height)
            cv2.line(frame, (sx, sy), (ex, ey), color, 2)

def label_prediction_by_side(side, model_stroke, coords):
    if side == "Right":
        return chord_labels[int(model_chord.predict([coords])[0])]

    return stroke_labels[int(model_stroke.predict([coords])[0])]

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
                score = detected_hand_data.handedness[i][0].score
                if score < MIN_DETECTION_CONFIDENCE:
                    continue  # ignore low-confidence detections

                draw_connections(frame, hand)

                coords = [v for lm in hand for v in (lm.x, lm.y)]
                side = detected_hand_data.handedness[i][0].category_name

                label = label_prediction_by_side(side, model_stroke, coords)

                if side == "Right":
                    current_chord_name = label["name"]
                else:
                    handle_stroke(label["name"])

                wrist = hand[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, label["name"], (cx, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Ghost Riff", frame)

        if cv2.waitKey(1) & 0xFF == ESC_KEY_ASCII_CODE:
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
