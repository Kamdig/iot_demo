import cv2
import time
import numpy as np
from flask import Response, stream_with_context

# Only load the TFLite bundle – NO TensorFlow imports here!
from thumbs_pi.assets import load_assets
from thumbs_pi.inference import classify_frame
from thumbs_pi.overlay import overlay_prediction

RTSP_URL = "rtsp://iotworldcam:smart123@192.168.1.204/stream2"
FRAME_SKIP = 3
MIN_CONFIDENCE = 0.6

# Load TFLite interpreter + metadata
bundle, class_names = load_assets(num_threads=4)


def gen_frames(gesture_callback=None):
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open RTSP stream: {RTSP_URL}")

    frame_idx = 0
    label, confidence, probs = None, 0.0, None

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        t1 = time.time()

        if not ret:
            time.sleep(0.1)
            continue

        frame_idx += 1

        # Only classify every FRAME_SKIP frames
        if frame_idx % FRAME_SKIP == 0:
            # Resize BEFORE sending to classify_frame()
            small_frame = cv2.resize(
                frame, (bundle.input_width, bundle.input_height)
            )

            t2 = time.time()
            pred_idx, confidence, probs = classify_frame(small_frame, bundle)
            t3 = time.time()

            print(
                f"⏱ Capture: {t1 - t0:.3f}s | "
                f"Resize: {t2 - t1:.3f}s | "
                f"Inference: {t3 - t2:.3f}s"
            )

            label = class_names[pred_idx]

            # Trigger callback if gesture confident
            if confidence >= MIN_CONFIDENCE and gesture_callback:
                gesture_callback(label)

        # Overlay prediction onto full frame
        if label is not None:
            overlay_prediction(
                frame,
                class_names=class_names,
                label=label,
                confidence=confidence,
                probabilities=probs,
                min_confidence=MIN_CONFIDENCE,
            )

        # Encode frame for MJPEG
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

    cap.release()


def mjpeg_response(gesture_callback=None):
    response = Response(
        stream_with_context(gen_frames(gesture_callback)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        direct_passthrough=False,
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
