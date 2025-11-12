import cv2
import time
import numpy as np
from flask import Response
from thumbs.assets import load_assets
from thumbs.inference import classify_frame
from thumbs.overlay import overlay_prediction

RTSP_URL = "rtsp://user:password@camera-ip:554/stream"
FRAME_SKIP = 3
MIN_CONFIDENCE = 0.6

# Load model and labels at import
bundle, class_names = load_assets()

def gen_frames():
    """Generator that yields annotated JPEG frames for Flask Response."""
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open RTSP stream: {RTSP_URL}")

    frame_idx = 0
    label, confidence, probs = None, 0.0, None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_idx += 1
        if frame_idx % FRAME_SKIP == 0:
            pred_idx, confidence, probs = classify_frame(frame, bundle)
            label = class_names[pred_idx]

        if label is not None:
            overlay_prediction(
                frame,
                class_names=class_names,
                label=label,
                confidence=confidence,
                probabilities=probs,
                min_confidence=MIN_CONFIDENCE,
            )

        # Convert frame to JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()

def mjpeg_response():
    """Return a Flask Response streaming MJPEG."""
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
