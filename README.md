# Ghost Riff

Play guitar chords in the air. Ghost Riff uses your webcam and a trained ML model to detect hand gestures in real time and play the corresponding chord audio. Left hand controls chord shape; right hand controls stroke type.

## Requirements

- Python 3.10+
- Webcam

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the MediaPipe hand landmark model and place it in the project root:

```bash
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Generate chord WAV files from the included base64 JSON sources:

```bash
python mix_chords.py
```

This decodes `sounds/base64-chords/*.json` → mixed WAVs in `sounds/chords/`. Requires `ffmpeg` on PATH.

Optional flags: `--folder`, `--output`, `--strum` (note delay in ms, default 27).

## Train your own models

```bash
# 1. Collect training images (repeat for each gesture class)
python collect_imgs.py --hand chord
python collect_imgs.py --hand stroke

# 2. Build datasets from collected images
python make_dataset.py --hand both

# 3. Train classifiers
python train_model.py --hand both
```

Labels must be kept in sync with class indices. Update `label_dicts` in `main.py` after adding or reordering classes.

## Run

```bash
source venv/bin/activate
python main.py
```

Press `ESC` to quit.

## Project structure

```
main.py                  — inference + audio playback
collect_imgs.py          — webcam-based image collection
make_dataset.py          — landmark extraction → pickle dataset
train_model.py           — classifier training
mix_chords.py            — decode base64 JSON → mixed chord WAVs
labels.py                — chord/stroke label definitions
hand_landmarker.task     — MediaPipe pre-trained landmark model (not in repo)
model_chord.pickle       — trained chord classifier (generated)
model_stroke.pickle      — trained stroke classifier (generated)
data/
  chord/                 — collected images, left hand
  stroke/                — collected images, right hand
sounds/
  base64-chords/         — source JSON files (base64-encoded MP3 notes per chord)
  chords/                — generated WAV files (output of mix_chords.py)
```
