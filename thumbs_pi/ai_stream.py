from thumbs_pi.overlay import overlay_prediction
from flask import Response, stream_with_context
from thumbs_pi.inference import classify_frame
from thumbs_pi.assets import load_assets
import time
import cv2
import os

RTSP_URL = os.environ.get("RTSP_URL") or os.environ.get("THUMBS_RTSP_URL")
if not RTSP_URL:
    raise RuntimeError(
        "RTSP_URL (or THUMBS_RTSP_URL) environment variable is required "
        "(e.g. rtsp://user:pass@host/path)"
    )
FRAME_SKIP = 3
MIN_CONFIDENCE = 0.6
NEUTRAL_LABEL = "neutral"
LAST_GESTURE = None
LAST_GESTURE_TIME = 0.0

# Load TFLite interpreter + metadata
bundle, class_names = load_assets(num_threads=4)

def gen_frames(gesture_callback=None):
    global LAST_GESTURE, LAST_GESTURE_TIME

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

        if frame_idx % FRAME_SKIP == 0:
            # Resize BEFORE sending to classify_frame()
            small_frame = cv2.resize(
                frame, (bundle.input_width, bundle.input_height)
            )

            t2 = time.time()
            pred_idx, confidence, probs = classify_frame(small_frame, bundle)
            t3 = time.time()

            up_idx = class_names.index("thumbs_up")
            down_idx = class_names.index("thumbs_down")

            up_p = float(probs[up_idx])
            down_p = float(probs[down_idx])

            print(
                f"probs: up={up_p:.3f}, down={down_p:.3f}, "
                f"label={class_names[pred_idx]}, conf={confidence:.3f}"
            )

            print(
                f"⏱ Capture: {t1 - t0:.3f}s | "
                f"Resize: {t2 - t1:.3f}s | "
                f"Inference: {t3 - t2:.3f}s"
            )

            # 🔁 New decision logic
            UP_THRESH = 0.55
            DOWN_THRESH = 0.45
            MARGIN = 0.05

            if up_p > UP_THRESH and up_p > down_p + MARGIN:
                label = "thumbs_up"
                confidence = up_p
            elif down_p > DOWN_THRESH and down_p > up_p + MARGIN:
                label = "thumbs_down"
                confidence = down_p
            else:
                label = NEUTRAL_LABEL


            # Debounce webhook: only send on change / cooldown, non-neutral
            if gesture_callback and confidence >= MIN_CONFIDENCE:
                now = time.time()
                if label != LAST_GESTURE:
                    gesture_callback(label)   # will send "thumbs_up", "thumbs_down", or "neutral"
                    LAST_GESTURE = label
                    LAST_GESTURE_TIME = time.time()


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
