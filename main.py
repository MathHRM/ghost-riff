import cv2
import mediapipe as mp

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

def draw_connections(frame, hand_landmarks):
    height, width, _ = frame.shape
    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        start = hand_landmarks[conn.start]
        end = hand_landmarks[conn.end]
        sx, sy = int(start.x * width), int(start.y * height)
        ex, ey = int(end.x * width), int(end.y * height)
        cv2.line(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)

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
            for hand in detected_hand_data.hand_landmarks:
                draw_connections(frame, hand)

        cv2.imshow("Hand Tracker (NEW API)", frame)

        if cv2.waitKey(1) & 0xFF == ESC_KEY_ASCII_CODE: # waitKey returns 32 bit int, so we need to do a bitwise AND with 0xFF to get the last 8 bits which represent the ASCII code of the key pressed
            break

cap.release()
cv2.destroyAllWindows()