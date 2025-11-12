"""
Standalone thumbs-up/-down detection utilities with optional Home Assistant integration.

This module coordinates helpers that live alongside it inside the thumbs_pi package:

  • thumbs_pi.assets          – model paths and asset loading
  • thumbs_pi.inference       – frame classification helpers
  • thumbs_pi.overlay         – OpenCV drawing utilities
  • thumbs_pi.home_assistant  – Home Assistant automation bridge
  • thumbs_pi.monitor         – RTSP stream loop

The public API mirrors the legacy thumbs module so existing imports can switch easily.
"""
from __future__ import annotations

import argparse
import logging
import os

from .assets import (
    CLASSES_PATH,
    CLASSES_TXT_PATH,
    MODEL_PATH,
    INFERENCE_TRANSFORM as TRANSFORM,
    TFLITE_CANDIDATES,
    TFLiteModelBundle,
    load_assets,
    load_class_names,
    save_class_names,
)
from .home_assistant import HAServiceAction, HomeAssistantGestureBridge, load_action_from_env
from .inference import classify_frame
from .monitor import run_rtsp_monitor
from .overlay import draw_probability_panel, overlay_prediction

__all__ = [
    "CLASSES_PATH",
    "CLASSES_TXT_PATH",
    "MODEL_PATH",
    "TRANSFORM",
    "TFLITE_CANDIDATES",
    "TFLiteModelBundle",
    "load_assets",
    "load_class_names",
    "save_class_names",
    "HAServiceAction",
    "HomeAssistantGestureBridge",
    "load_action_from_env",
    "classify_frame",
    "draw_probability_panel",
    "overlay_prediction",
    "run_rtsp_monitor",
    "parse_args",
    "main",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thumbs-up/down detection with optional Home Assistant actions.")
    parser.add_argument(
        "--rtsp-url",
        default=os.getenv("THUMBS_RTSP_URL", "rtsp://iotworldcam:smart123@10.136.171.24/stream2"),
        help="RTSP stream URL to connect to.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=int(os.getenv("THUMBS_FRAME_SKIP", "2")),
        help="Process every Nth frame to reduce load (default via THUMBS_FRAME_SKIP or 2).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=float(os.getenv("THUMBS_MIN_CONFIDENCE", "0.6")),
        help="Minimum confidence required to report a gesture (default via THUMBS_MIN_CONFIDENCE or 0.6).",
    )
    parser.add_argument(
        "--action-cooldown",
        type=float,
        default=float(os.getenv("THUMBS_ACTION_COOLDOWN", "2.0")),
        help="Seconds to wait before repeating the same Home Assistant action (default 2).",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without displaying the OpenCV preview window.",
    )
    parser.add_argument(
        "--disable-ha",
        action="store_true",
        help="Disable Home Assistant service calls even if configured.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("THUMBS_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    run_rtsp_monitor(
        rtsp_url=args.rtsp_url,
        frame_skip=args.frame_skip,
        min_confidence=args.min_confidence,
        display=not args.no_window,
        action_cooldown=args.action_cooldown,
        enable_home_assistant=not args.disable_ha,
    )


if __name__ == "__main__":
    main()
