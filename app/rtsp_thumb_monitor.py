"""
Real-time thumbs detector with on-screen visualization only.

Uses shared utilities from ``thumbs_pi.thumbs`` to load the model, classify
frames from the configured RTSP stream, and render probability overlays. This
script is useful for monitoring predictions without triggering Home Assistant
actions (see ``thumbs_pi/thumbs.py`` for automation support).
"""
from __future__ import annotations

from thumbs_pi.thumbs import classify_frame, load_assets, overlay_prediction
from typing import Optional
import argparse
import time
import sys
import cv2


# Parse CLI flags for connecting to and monitoring the RTSP stream.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview thumbs detection results from an RTSP stream.")
    parser.add_argument(
        "--rtsp-url",
        default="rtsp://iotworldcam:smart123@192.168.1.204/stream2",
        help="RTSP stream URL to connect to.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Process every Nth frame to reduce load (default: 2).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence required to log a detection (default: 0.6).",
    )
    return parser.parse_args()


# Run the monitoring loop with periodic classification and logging.
def main() -> None:
    args = parse_args()

    try:
        bundle, class_names = load_assets()
    except Exception as exc:
        print(f"[ERROR] Failed to load model assets: {exc}")
        sys.exit(1)

    print("[INFO] Model loaded (TFLite).")
    print(f"[INFO] Class labels: {list(class_names)}")

    cap = cv2.VideoCapture(args.rtsp_url)
    # Fail fast when the RTSP connection cannot be established.
    if not cap.isOpened():
        print(f"[ERROR] Unable to open RTSP stream: {args.rtsp_url}")
        sys.exit(1)

    print("[INFO] Press 'q' to exit.")
    frame_idx = 0
    last_label: Optional[str] = None
    last_report_time = 0.0

    try:
        # Continuously read frames until the user quits.
        while True:
            ret, frame = cap.read()
            # Retry briefly when a frame cannot be read from the stream.
            if not ret:
                print("[WARNING] Failed to read frame from stream. Retrying...")
                time.sleep(0.5)
                continue

            frame_idx += 1
            # Only classify every Nth frame to control compute usage.
            if frame_idx % max(args.frame_skip, 1) != 0:
                cv2.imshow("Thumbs Detector", frame)
                # Exit early if the user presses "q".
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            predicted_idx, confidence, probabilities = classify_frame(frame, bundle)
            label = class_names[predicted_idx]

            certainty = f"{confidence * 100:.1f}%"
            now = time.time()
            # Log the detection when it meets the confidence and cooldown rules.
            if confidence >= args.min_confidence and (label != last_label or now - last_report_time > 2.0):
                breakdown = ", ".join(
                    f"{name}: {prob * 100:.1f}%"
                    for name, prob in zip(class_names, probabilities)
                )
                print(f"[{time.strftime('%H:%M:%S')}] Detected {label} ({certainty}) | {breakdown}")
                last_label = label
                last_report_time = now

            overlay_prediction(
                frame,
                class_names=class_names,
                label=label,
                confidence=confidence,
                probabilities=probabilities,
                min_confidence=args.min_confidence,
            )
            cv2.imshow("Thumbs Detector", frame)

            # Allow the operator to exit the preview loop via "q".
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Support running this module directly as a script.
    main()
