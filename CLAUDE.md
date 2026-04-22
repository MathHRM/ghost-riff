# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run

```bash
source venv/bin/activate
python main.py
```

Requires webcam, `hand_landmarker.task`, `model_chord.pickle`, `model_stroke.pickle`. Press ESC to quit.

## Stack

- **MediaPipe Tasks API** (`mp.tasks.vision`) — hand landmark detection. Uses new Tasks API, not legacy `Hands()` — incompatible APIs, do not mix.
- **OpenCV** — webcam capture, frame display, drawing
- **scikit-learn** — RandomForestClassifier for gesture classification
- **pygame** — chord audio playback
- **mutagen** — reads MP3 duration to control playback length
- **pydub** — mix/overlay audio segments; requires `ffmpeg` on PATH for MP3 decoding

## Architecture

Two independent classifiers: one per hand. `main.py` is inference entry point.

### Training pipeline

```
collect_imgs.py --hand {chord|stroke} [--class N] [--duration S]
make_dataset.py --hand {chord|stroke|both}
train_model.py  --hand {chord|stroke|both}
mix_chords.py   <folder>   ← decode base64 MP3 JSONs → mixed chord WAVs
```

Outputs: `data/{hand}/{class_idx}/frame_*.jpg` → `dataset_{hand}.pickle` → `model_{hand}.pickle`

### Data structure

```
data/
  chord/    ← left hand
    0/
    1/
    ...
  stroke/   ← right hand
    0/
    1/
    ...
sounds/
  chords/   ← MP3 files, named by chord label (e.g. C.mp3)
```

## Key constraints

- MediaPipe Tasks API only — `HandLandmarkerOptions`, `VisionRunningMode`
- Each hand → 42 features (21 landmarks × x,y). Models must be trained with this exact shape.
- `make_dataset.py` uses `VisionRunningMode.IMAGE` (not VIDEO) — static frame processing, different init path than inference
- Frame is horizontally flipped before MediaPipe: MediaPipe's Left/Right is inverted relative to user. This is intentional — do not remove the flip.
- `label_dicts` in `main.py` must stay in sync with class indices used during training

## Self-improvement

After any code change, update this file to reflect new architecture, constraints, or stack additions. Keep sections accurate — stale docs are worse than none.
