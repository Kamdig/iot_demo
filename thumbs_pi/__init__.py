"""Standalone thumbs-up/-down detection toolkit with optional Home Assistant automation."""
from __future__ import annotations

from .assets import (
    BASE_DIR,
    CLASSES_PATH,
    CLASSES_TXT_PATH,
    MODEL_PATH,
    INFERENCE_TRANSFORM,
    TFLITE_CANDIDATES,
    TFLiteModelBundle,
    load_assets,
    load_class_names,
    save_class_names,
)
from .home_assistant import (
    HAServiceAction,
    HomeAssistantClient,
    HomeAssistantGestureBridge,
    get_client,
    load_action_from_env,
    parse_service_string,
    set_client_factory,
)
from .inference import classify_frame
from .monitor import run_rtsp_monitor
from .overlay import draw_probability_panel, overlay_prediction
from .thumbs import main, parse_args


TRANSFORM = INFERENCE_TRANSFORM

__all__ = [
    "BASE_DIR",
    "CLASSES_PATH",
    "CLASSES_TXT_PATH",
    "MODEL_PATH",
    "TFLITE_CANDIDATES",
    "TFLiteModelBundle",
    "INFERENCE_TRANSFORM",
    "HAServiceAction",
    "HomeAssistantClient",
    "HomeAssistantGestureBridge",
    "TRANSFORM",
    "classify_frame",
    "draw_probability_panel",
    "get_client",
    "load_action_from_env",
    "load_assets",
    "load_class_names",
    "main",
    "overlay_prediction",
    "parse_args",
    "parse_service_string",
    "run_rtsp_monitor",
    "save_class_names",
    "set_client_factory",
]
