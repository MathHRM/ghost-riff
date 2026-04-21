# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run

```bash
source venv/bin/activate
python main.py
```

Requires webcam. Press ESC to quit.

## Stack

- **MediaPipe** (Tasks API, `VisionRunningMode.VIDEO`) — hand landmark detection via `hand_landmarker.task` model file
- **OpenCV** — webcam capture, frame display, drawing

## Architecture

Single-file app (`main.py`). Pipeline:

1. OpenCV captures frame from webcam
2. Frame flipped horizontally (mirror effect), converted BGR→RGB
3. Wrapped in `mp.Image`, fed to `HandLandmarker.detect_for_video` with timestamp from `CAP_PROP_POS_MSEC`
4. Detected landmarks drawn as green circles (positions are % of frame, converted to pixels)

Key constraint: MediaPipe Tasks API (`mp.tasks.vision`) differs from legacy `mp.solutions` API — uses `HandLandmarkerOptions`, not `Hands()`.

## Self-improvement

After any code change, update this file to reflect new architecture, constraints, or stack additions. Keep sections accurate — stale docs are worse than none.
