# meteorite-detector-video

Drone video processing pipeline for meteorite candidate detection.

Part of the [MeteorAI](https://github.com/charleslucas/meteorai) project.

## Overview

Takes drone survey video as input, runs the trained YOLOv8 model frame-by-frame,
and outputs timestamped candidate detections for ground follow-up.

## Structure

```
meteorite-detector-video/
  processor/       - Core video processing and detection logic
  ui/              - Flask web UI for running and reviewing detections
  models/          - Trained model weights (stored via Git LFS)
  output/          - Detection results (gitignored)
```

## Usage

```bash
# Launch the Flask UI
python ui/app.py

# Process a video file directly
python processor/detect.py --video path/to/survey.mp4
```

## Requirements

**System prerequisites:**
- [Git LFS](https://git-lfs.com) — required to download the model weights (`models/best.pt`).
  Install it, then run `git lfs pull` inside this directory if the model file is missing.
  Without it, `models/best.pt` will be a small text pointer file instead of the actual model.

**Python packages:** see `requirements.txt`

```bash
pip install -r requirements.txt
```
