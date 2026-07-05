"""
Meteorite Detector Video — Flask UI
Run: python ui/app.py
"""
import json
import os
import queue
import sys
import tempfile
import threading
import webbrowser
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

UI_DIR    = Path(__file__).resolve().parent
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
JOBS: dict = {}

app = Flask(__name__, static_folder=str(UI_DIR), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(UI_DIR, "detector.html")


@app.route("/api/model-info")
def model_info():
    meta = MODELS_DIR / "model_info.json"
    models = sorted(MODELS_DIR.glob("*.pt")) if MODELS_DIR.exists() else []
    info = json.loads(meta.read_text()) if meta.exists() else {}
    info["models"] = [m.name for m in models]
    return jsonify(info)


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    conf           = float(request.form.get("conf", 0.25))
    frame_step     = int(request.form.get("frame_step", 5))
    iou            = float(request.form.get("iou", 0.5))
    temporal_window = int(request.form.get("temporal_window", 15))
    model_name     = request.form.get("model", "best.pt")

    suffix = Path(f.filename or "video").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.save(tmp.name)
    tmp.close()

    job_id = uuid.uuid4().hex[:8]
    q = queue.Queue()
    JOBS[job_id] = {"queue": q, "results": None, "error": None}

    threading.Thread(
        target=_run_detection,
        args=(job_id, tmp.name, model_name, conf, frame_step, iou, temporal_window),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id})


def _run_detection(job_id, video_path, model_name, conf, frame_step, iou, temporal_window):
    q = JOBS[job_id]["queue"]
    try:
        from processor.detect import MeteoriteDetector

        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        detector = MeteoriteDetector(
            model_path=str(model_path),
            conf_threshold=conf,
            frame_step=frame_step,
            iou_threshold=iou,
            temporal_window=temporal_window,
        )

        def progress(frame_num, total):
            pct = min(frame_num / max(total, 1) * 85, 85)
            q.put({"type": "progress", "pct": round(pct, 1),
                   "frame": frame_num, "total": total})

        candidates = detector.process_video(video_path, progress_callback=progress)

        JOBS[job_id]["results"] = candidates
        q.put({"type": "complete", "candidates": candidates})

    except Exception as exc:
        JOBS[job_id]["error"] = str(exc)
        q.put({"type": "error", "message": str(exc)})
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass


@app.route("/api/stream/<job_id>")
def stream(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "Unknown job"}), 404

    def generate():
        q = JOBS[job_id]["queue"]
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("complete", "error"):
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Meteorite Detector Video UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    print(f"\nMeteoriteDetector UI  →  {url}\n")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
