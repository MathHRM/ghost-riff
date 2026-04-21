# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run

```bash
source venv/bin/activate
python main.py
```

Requires webcam and both `model_chord.pickle` + `model_stroke.pickle`. Press ESC to quit.

## Stack

- **MediaPipe** (Tasks API, `VisionRunningMode.VIDEO`) — hand landmark detection via `hand_landmarker.task` model file
- **OpenCV** — webcam capture, frame display, drawing
- **scikit-learn** — RandomForestClassifier for gesture classification

## Architecture

Dual-model gesture recognition pipeline. `main.py` is the entry point.

### Inference (`main.py`)
1. OpenCV captures frame
2. Frame flipped horizontally (mirror), converted BGR→RGB
3. Wrapped in `mp.Image`, fed to `HandLandmarker.detect_for_video`
4. Each detected hand is drawn + classified independently:
   - `handedness[i].category_name == "Right"` → user's **left hand** → chord model
   - `handedness[i].category_name == "Left"` → user's **right hand** → stroke model
5. Predicted label drawn near each wrist

**Handedness caveat:** frame is flipped before MediaPipe, so MediaPipe's Left/Right is inverted relative to the user.

### Training pipeline

```
collect_imgs.py --hand {chord|stroke}   → data/{hand}/{class_idx}/frame_*.jpg
make_dataset.py --hand {chord|stroke}   → dataset_{hand}.pickle
train_model.py  --hand {chord|stroke}   → model_{hand}.pickle
```

### Data structure

```
data/
  chord/    ← left hand, chord shape classes
    0/
    1/
    ...
  stroke/   ← right hand, stroke type classes
    0/
    1/
    ...
```

### Label dicts (update in `main.py` as classes grow)

```python
chord_labels  = {0: "C", 1: "G", 2: "Am"}
stroke_labels = {0: "Down", 1: "Up", 2: "Mute"}
```

## Key constraints

- MediaPipe Tasks API (`mp.tasks.vision`) — uses `HandLandmarkerOptions`, not legacy `Hands()`
- Each hand produces 42 features (21 landmarks × x,y) — models expect exactly 42
- `make_dataset.py` uses `VisionRunningMode.IMAGE` (not VIDEO) for static frame processing

## Self-improvement

After any code change, update this file to reflect new architecture, constraints, or stack additions. Keep sections accurate — stale docs are worse than none.
