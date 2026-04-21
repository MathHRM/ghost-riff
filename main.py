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

model_dict = pickle.load(open('./model.pickle', 'rb'))
model = model_dict['model']
labels_dict = {0: 'Joia', 1: 'Paz', 2: 'De boa'}

FINGER_COLORS = [
    (HandLandmarksConnections.HAND_PALM_CONNECTIONS,         (255, 255, 255)),  # white
    (HandLandmarksConnections.HAND_THUMB_CONNECTIONS,        (0, 128, 255)),    # orange
    (HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS, (0, 255, 0)),      # green
    (HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS,(255, 0, 0)),      # blue
    (HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS,  (0, 0, 255)),      # red
    (HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS, (255, 0, 255)),    # magenta
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

        # converte para RGB
        frame = cv2.flip(frame, 1) # flip horizontalmente para ficar como um espelho
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detected_hand_data = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        if detected_hand_data.hand_landmarks:
            h, w, _ = frame.shape
            for hand in detected_hand_data.hand_landmarks:
                draw_connections(frame, hand)

                coords = []
                for landmark in hand:
                    coords.append(landmark.x)
                    coords.append(landmark.y)

                prediction = model.predict([np.asarray(coords)])
                predicted_label = labels_dict[int(prediction[0])]

                wrist = hand[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, predicted_label, (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Hand Tracker (NEW API)", frame)

        if cv2.waitKey(1) & 0xFF == ESC_KEY_ASCII_CODE: # waitKey returns 32 bit int, so we need to do a bitwise AND with 0xFF to get the last 8 bits which represent the ASCII code of the key pressed
            break

cap.release()
cv2.destroyAllWindows()