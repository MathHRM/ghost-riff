import os
import glob
import pickle
import argparse
import cv2
import mediapipe as mp

parser = argparse.ArgumentParser()
parser.add_argument("--hand", choices=["chord", "stroke"], required=True)
args = parser.parse_args()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

data = []
labels = []

with HandLandmarker.create_from_options(options) as landmarker:
    for class_dir in glob.glob(f"data/{args.hand}/*/"):
        label = int(os.path.basename(os.path.normpath(class_dir)))
        image_paths = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))

        for img_path in image_paths:
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                continue

            coords = []
            for lm in result.hand_landmarks[0]:
                coords.append(lm.x)
                coords.append(lm.y)

            data.append(coords)
            labels.append(label)

        print(f"class {label}: {sum(1 for l in labels if l == label)} samples")

output = f"dataset_{args.hand}.pickle"
with open(output, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print(f"\n{output} written — {len(data)} total samples, {len(set(labels))} classes")
