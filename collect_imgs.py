import cv2
import os
import time
import glob

ESC_KEY = 27
COLLECT_DURATION = 5.0


def next_class_index():
    existing = glob.glob("data/*/")
    if not existing:
        return 0
    indices = []
    for path in existing:
        name = os.path.basename(os.path.normpath(path))
        try:
            indices.append(int(name))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def process_frame(frame, state, class_idx, frame_count, collect_start):
    if state == "IDLE":
        cv2.putText(frame, f"IDLE  press Q to record  (class {class_idx})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        return state, class_idx, frame_count, collect_start

    elapsed = time.time() - collect_start
    remaining = COLLECT_DURATION - elapsed

    if remaining <= 0:
        return "IDLE", class_idx + 1, 0, None

    cv2.imwrite(f"data/{class_idx}/frame_{frame_count:04d}.jpg", frame)
    frame_count += 1
    cv2.putText(frame, f"REC  {remaining:.1f}s  [{frame_count} frames]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return "COLLECTING", class_idx, frame_count, collect_start


cap = cv2.VideoCapture(0)

state = "IDLE"
class_idx = next_class_index()
frame_count = 0
collect_start = None

os.makedirs("data", exist_ok=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    key = cv2.waitKey(1) & 0xFF

    if key == ESC_KEY:
        break

    if key == ord('q') and state == "IDLE":
        state = "COLLECTING"
        collect_start = time.time()
        os.makedirs(f"data/{class_idx}", exist_ok=True)
        frame_count = 0

    state, class_idx, frame_count, collect_start = process_frame(
        frame, state, class_idx, frame_count, collect_start
    )

    cv2.imshow("Collector", frame)

cap.release()
cv2.destroyAllWindows()
