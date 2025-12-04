from __future__ import annotations
from .overlay import draw_probability_panel, overlay_prediction
from .inference import classify_frame
from .monitor import run_rtsp_monitor
import argparse
import logging
import os
from .assets import (
    CLASS_NAMES_PATH,
    TFLITE_QUANT_PATH,
    TFLiteModelBundle,
    load_assets,
)
from app.homeassistant.client import HAServiceAction, HomeAssistantGestureBridge, load_action_from_env

__all__ = [
    "CLASS_NAMES_PATH",
    "TFLITE_QUANT_PATH",
    "TFLiteModelBundle",
    "load_assets",
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


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thumbs-up/down detection with optional Home Assistant actions."
    )
    parser.add_argument(
        "--rtsp-url",
        default=os.getenv("RTSP_URL") or os.getenv("THUMBS_RTSP_URL"),
        help="RTSP stream URL to connect to.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=int(os.getenv("THUMBS_FRAME_SKIP", "2")),
        help="Process every Nth frame (default 2).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=float(os.getenv("THUMBS_MIN_CONFIDENCE", "0.6")),
        help="Minimum confidence to report a gesture (default 0.6).",
    )
    parser.add_argument(
        "--action-cooldown",
        type=float,
        default=float(os.getenv("THUMBS_ACTION_COOLDOWN", "2.0")),
        help="Cooldown between repeated actions (default 2s).",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable OpenCV display window.",
    )
    parser.add_argument(
        "--disable-ha",
        action="store_true",
        help="Disable Home Assistant service actions.",
    )
    args = parser.parse_args(args)
    if not args.rtsp_url:
        parser.error("RTSP URL required via --rtsp-url or RTSP_URL/THUMBS_RTSP_URL environment variable.")
    return args


def main() -> None:
    logging.basicConfig(
        level=os.getenv("THUMBS_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args()

    # Load TFLite model + class names
    bundle, class_names = load_assets()

    run_rtsp_monitor(
        rtsp_url=args.rtsp_url,
        frame_skip=args.frame_skip,
        min_confidence=args.min_confidence,
        display=not args.no_window,
        action_cooldown=args.action_cooldown,
        enable_home_assistant=not args.disable_ha,
        bundle=bundle,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
