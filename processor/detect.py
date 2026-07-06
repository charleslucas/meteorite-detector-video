"""
MeteoriteDetector - YOLOv8-based drone video processing.

Processes video frame-by-frame, runs the trained model, deduplicates
detections across frames, and outputs a JSON candidate list.

Usage:
    from processor.detect import MeteoriteDetector
    detector = MeteoriteDetector()
    candidates = detector.process_video("survey.mp4", output_path="annotated.mp4")
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = MODELS_DIR / "best.pt"


class MeteoriteDetector:
    METEORITE_PROFILES = {
        "general": {"name": "General"},
        "iron":    {"name": "Iron"},
        "stony":   {"name": "Stony"},
    }

    def __init__(
        self,
        meteorite_type: str = "general",
        model_path: str | None = None,
        conf_threshold: float = 0.25,
        frame_step: int = 5,
        iou_threshold: float = 0.5,
        temporal_window: int = 60,
    ):
        """
        Args:
            meteorite_type:   "general" | "iron" | "stony" (reserved for future model variants)
            model_path:       Path to .pt file. Defaults to models/best.pt.
            conf_threshold:   Minimum YOLOv8 confidence to keep a detection.
            frame_step:       Process every Nth frame (5 = 6fps from 30fps source).
            iou_threshold:    IoU overlap to consider two frame-detections the same object.
            temporal_window:  Frames to look back when deduplicating across time.
        """
        from ultralytics import YOLO

        profile = self.METEORITE_PROFILES.get(meteorite_type, self.METEORITE_PROFILES["general"])
        self.meteorite_type = meteorite_type
        self.conf_threshold = conf_threshold
        self.frame_step = frame_step
        self.iou_threshold = iou_threshold
        self.temporal_window = temporal_window

        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run 'release_model.py' from the parent meteorai repo to install a model."
            )
        self.model = YOLO(str(path))
        self.model_path = path

        # Load model metadata if present
        meta_path = path.parent / "model_info.json"
        self.model_info = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        print(f"MeteoriteDetector ready")
        print(f"  Profile:    {profile['name']}")
        print(f"  Model:      {path.name}")
        print(f"  Classes:    {list(self.model.names.values())}")
        print(f"  Confidence: {conf_threshold}")
        print(f"  Frame step: every {frame_step} frames")
        if self.model_info:
            print(f"  Released:   {self.model_info.get('released_at', '?')}")
            print(f"  mAP50:      {self.model_info.get('map50', '?')}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(self, video_path, output_path=None, progress_callback=None, stop_callback=None):
        """
        Process a video file and return a list of deduplicated candidate detections.

        Args:
            video_path:         Path to input video.
            output_path:        Optional path for annotated output video.
            progress_callback:  Optional callable(frame, total) for UI progress.

        Returns:
            List of candidate dicts:
                {
                    "first_frame": int,
                    "last_frame":  int,
                    "timestamp_s": float,   # time in video of first detection
                    "bbox":        [x, y, w, h],  # pixel coords in original frame
                    "label":       str,
                    "confidence":  float,
                    "frame_count": int,     # how many frames this was seen in
                }
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            writer = self._make_writer(output_path, fps, width, height)

        print(f"Video: {video_path.name}  {width}x{height}  {fps:.0f}fps  {total} frames")
        print(f"Processing ~{total // self.frame_step} sampled frames...")

        candidates = []   # running deduplicated list
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if stop_callback and stop_callback():
                print("  Detection stopped by user.")
                break

            if frame_num % self.frame_step == 0:
                hits = self._run_inference(frame, frame_num, fps)
                self._merge_hits(candidates, hits)

                if writer is not None:
                    self._draw_detections(frame, hits)

                if progress_callback:
                    progress_callback(frame_num, total, frame, hits, candidates)
                elif frame_num % (self.frame_step * 30) == 0:
                    print(f"  Frame {frame_num}/{total}  candidates so far: {len(candidates)}")

            if writer is not None:
                writer.write(frame)

            frame_num += 1

        cap.release()
        if writer:
            writer.release()

        print(f"\nComplete: {len(candidates)} candidates")
        return candidates

    def save_results(self, candidates, video_path, output_dir=None):
        """Save candidates to a JSON file alongside the video (or in output_dir)."""
        video_path = Path(video_path)
        out_dir = Path(output_dir) if output_dir else video_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / (video_path.stem + "_detections.json")
        payload = {
            "video":      video_path.name,
            "model":      self.model_path.name,
            "model_info": self.model_info,
            "conf":       self.conf_threshold,
            "frame_step": self.frame_step,
            "candidates": candidates,
        }
        json_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved: {json_path}")
        return json_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inference(self, frame, frame_num, fps):
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        hits = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                hits.append({
                    "frame":      frame_num,
                    "timestamp_s": round(frame_num / fps, 2),
                    "bbox":       [x1, y1, x2 - x1, y2 - y1],
                    "label":      self.model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                })
        return hits

    def _draw_detections(self, frame, hits):
        for det in hits:
            x, y, w, h = det["bbox"]
            conf = det["confidence"]
            label = det["label"]
            color = (0, 255, 0) if conf >= 0.5 else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {conf:.0%}", (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _make_writer(self, output_path, fps, width, height):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Try platform-appropriate codecs
        for fourcc_str in ("avc1", "mp4v", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if writer.isOpened():
                return writer
        raise RuntimeError("Could not open a VideoWriter with any available codec.")

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(ax, bx);  iy = max(ay, by)
        ix2 = min(ax+aw, bx+bw);  iy2 = min(ay+ah, by+bh)
        if ix2 <= ix or iy2 <= iy:
            return 0.0
        inter = (ix2-ix) * (iy2-iy)
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0.0

    def _merge_hits(self, candidates, hits):
        """Merge a frame's hits into the running candidates list (incremental dedup)."""
        for det in hits:
            merged = False
            for cand in candidates:
                if (det["frame"] - cand["last_frame"] <= self.temporal_window
                        and det["label"] == cand["label"]
                        and self._iou(det["bbox"], cand["bbox"]) >= self.iou_threshold):
                    if det["confidence"] > cand["confidence"]:
                        cand["bbox"]       = det["bbox"]
                        cand["confidence"] = det["confidence"]
                    cand["last_frame"]  = det["frame"]
                    cand["frame_count"] += 1
                    merged = True
                    break
            if not merged:
                candidates.append({
                    "first_frame": det["frame"],
                    "last_frame":  det["frame"],
                    "timestamp_s": det["timestamp_s"],
                    "bbox":        det["bbox"],
                    "label":       det["label"],
                    "confidence":  det["confidence"],
                    "frame_count": 1,
                })
