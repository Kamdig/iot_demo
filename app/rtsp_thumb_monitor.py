"""
Real-time thumbs detector with on-screen visualization only.

Uses shared utilities from ``thumbs_ai.thumbs`` to load the model, classify
frames from the configured RTSP stream, and render probability overlays. This
script is useful for monitoring predictions without triggering Home Assistant
actions (see ``thumbs_ai/thumbs.py`` for automation support).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import cv2
from thumbs_ai.thumbs import classify_frame, load_assets, overlay_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview thumbs detection results from an RTSP stream.")
    parser.add_argument(
        "--rtsp-url",
        default="rtsp://iotworldcam:smart123@10.136.171.24/stream2",
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


def main() -> None:
    args = parse_args()

    try:
        model, class_names, device = load_assets()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Failed to load model assets: {exc}")
        sys.exit(1)

    print(f"[INFO] Loaded model on device: {device}")
    print(f"[INFO] Class labels: {list(class_names)}")

    cap = cv2.VideoCapture(args.rtsp_url)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open RTSP stream: {args.rtsp_url}")
        sys.exit(1)

    print("[INFO] Press 'q' to exit.")
    frame_idx = 0
    last_label: Optional[str] = None
    last_report_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame from stream. Retrying...")
                time.sleep(0.5)
                continue

            frame_idx += 1
            if frame_idx % max(args.frame_skip, 1) != 0:
                cv2.imshow("Thumbs Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            predicted_idx, confidence, probabilities = classify_frame(frame, model, device)
            label = class_names[predicted_idx]

            certainty = f"{confidence * 100:.1f}%"
            now = time.time()
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

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
