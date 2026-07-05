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
  ui/              - Streamlit interface for running and reviewing detections
  models/          - Symlink or pointer to trained model weights
  output/          - Detection results (gitignored)
```

## Usage

```bash
# Process a video file
python processor/detect.py --video path/to/survey.mp4

# Launch the Streamlit UI
streamlit run ui/app.py
```

## Requirements

See `requirements.txt`. Shares the YOLOv8 model with the parent meteorai repo.
