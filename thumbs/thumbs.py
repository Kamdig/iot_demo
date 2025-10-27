"""
Thumbs-up/-down detection utilities with optional Home Assistant integration.

This module can be executed as a script to watch an RTSP stream, render live
predictions, and trigger Home Assistant services when confident gestures are
observed. It also exports helper functions that other scripts (such as
app/rtsp_thumb_monitor.py) can reuse for model loading and inference.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import json
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from app.homeassistant.client import get_client


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
CLASSES_PATH = BASE_DIR / "class_names.pkl"


class ConvNeXtClassifier(nn.Module):
    """Classifier wrapper around ConvNeXt Tiny with a custom head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        for param in self.convnext.parameters():
            param.requires_grad = False
        self.convnext.classifier[2] = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnext(x)


def load_assets() -> Tuple[ConvNeXtClassifier, Sequence[str], torch.device]:
    """Load the model weights and class names required for inference."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. Train the model to generate 'best_model.pth'."
        )
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found at {CLASSES_PATH}. Train the model to generate 'class_names.pkl'."
        )

    with CLASSES_PATH.open("rb") as class_file:
        class_names = pickle.load(class_file)

    if not isinstance(class_names, (list, tuple)):
        raise TypeError(f"Expected class names to be a list/tuple, got {type(class_names)}")
    class_names = list(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtClassifier(num_classes=len(class_names)).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, class_names, device


TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def classify_frame(
    frame: np.ndarray,
    model: ConvNeXtClassifier,
    device: torch.device,
) -> Tuple[int, float, np.ndarray]:
    """Run a forward pass on the provided BGR OpenCV frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    return predicted_idx.item(), confidence.item(), probabilities.cpu().numpy()


def draw_probability_panel(
    frame: np.ndarray,
    class_names: Sequence[str],
    probabilities: np.ndarray,
    *,
    origin: Tuple[int, int] = (10, 70),
    width: int = 280,
    line_height: int = 28,
) -> None:
    """Overlay per-class probability bars onto the frame."""
    if len(class_names) == 0:
        return

    x_start, y_start = origin
    padding = 8
    total_height = line_height * len(class_names) + padding

    panel = frame.copy()
    cv2.rectangle(
        panel,
        (x_start - padding, y_start - line_height),
        (x_start + width, y_start + total_height),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.addWeighted(panel, 0.4, frame, 0.6, 0, frame)

    bar_max_width = width - 150
    bar_start_x = x_start + 110
    score_x = bar_start_x + bar_max_width + 10

    top_idx = int(np.argmax(probabilities))
    for idx, (name, prob) in enumerate(zip(class_names, probabilities)):
        line_y = y_start + idx * line_height
        bar_length = int(bar_max_width * float(prob))
        bar_color = (0, 200, 0) if idx == top_idx else (60, 60, 255)
        cv2.rectangle(
            frame,
            (bar_start_x, line_y - 18),
            (bar_start_x + max(bar_length, 1), line_y - 4),
            bar_color,
            thickness=-1,
        )
        label = f"{name:<12}"
        value = f"{prob * 100:5.1f}%"
        cv2.putText(frame, label, (x_start, line_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, value, (score_x, line_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def overlay_prediction(
    frame: np.ndarray,
    class_names: Sequence[str],
    label: str,
    confidence: float,
    probabilities: np.ndarray,
    *,
    min_confidence: float,
) -> None:
    """Render the active prediction and probability histogram on the frame."""
    certainty = f"{confidence * 100:.1f}%"
    if confidence < min_confidence:
        label_display = "uncertain"
        text_color = (0, 255, 255)
    else:
        label_display = label
        text_color = (0, 255, 0) if label == "thumbs_up" else (0, 0, 255)

    overlay = f"{label_display} ({certainty})"
    cv2.putText(
        frame,
        overlay,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        text_color,
        2,
        cv2.LINE_AA,
    )
    draw_probability_panel(frame, class_names, probabilities)


@dataclass
class HAServiceAction:
    """Wrapper around a Home Assistant service invocation."""

    domain: str
    service: str
    payload: Dict[str, Any]

    def execute(self) -> bool:
        client = get_client()
        if client is None:
            logger.error(
                "Home Assistant client unavailable; cannot execute %s.%s with payload %s",
                self.domain,
                self.service,
                self.payload,
            )
            return False

        success = client.call_service(self.domain, self.service, self.payload)
        if success:
            logger.info("Home Assistant service %s.%s executed with %s", self.domain, self.service, self.payload)
        else:
            logger.error("Home Assistant service %s.%s failed for payload %s", self.domain, self.service, self.payload)
        return success


def parse_service_string(service: str) -> Tuple[str, str]:
    """Split a Home Assistant service string of the form 'domain.service'."""
    if "." not in service:
        raise ValueError(f"Service '{service}' must be in the format 'domain.service'.")
    domain, service_name = service.split(".", 1)
    if not domain or not service_name:
        raise ValueError(f"Service '{service}' must be in the format 'domain.service'.")
    return domain, service_name


def load_action_from_env(
    prefix: str,
    default_service: Optional[str],
    default_payload: Optional[Dict[str, Any]],
) -> Optional[HAServiceAction]:
    """
    Build an HAServiceAction from environment variables, falling back to defaults.

    Expected environment keys:
      <prefix>_SERVICE  (e.g., 'HA_THUMBS_UP_SERVICE' -> 'light.turn_on')
      <prefix>_PAYLOAD  (JSON string merged into the payload)
    """
    service_key = f"{prefix}_SERVICE"
    payload_key = f"{prefix}_PAYLOAD"

    service_value = os.getenv(service_key, (default_service or "")).strip()
    if not service_value:
        logger.debug("%s not configured; skipping service action.", service_key)
        return None

    payload: Dict[str, Any] = {}
    if default_payload:
        payload.update(default_payload)

    payload_override = os.getenv(payload_key)
    if payload_override:
        try:
            payload.update(json.loads(payload_override))
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON for %s: %s", payload_key, exc)

    try:
        domain, service_name = parse_service_string(service_value)
    except ValueError as exc:
        logger.error("Skipping Home Assistant action for %s: %s", service_key, exc)
        return None

    return HAServiceAction(domain=domain, service=service_name, payload=payload)


@dataclass
class HomeAssistantGestureBridge:
    """Coordinate gesture detections with Home Assistant service calls."""

    min_confidence: float
    cooldown_seconds: float
    thumbs_up_action: Optional[HAServiceAction] = None
    thumbs_down_action: Optional[HAServiceAction] = None
    _last_triggered: Dict[str, float] = field(default_factory=dict, init=False)

    def handle(self, label: str, confidence: float) -> None:
        if confidence < self.min_confidence:
            return

        action = self._select_action(label)
        if action is None:
            return

        now = time.time()
        last_fired = self._last_triggered.get(label, 0.0)
        if now - last_fired < self.cooldown_seconds:
            return

        if action.execute():
            self._last_triggered[label] = now

    def _select_action(self, label: str) -> Optional[HAServiceAction]:
        if label == "thumbs_up":
            return self.thumbs_up_action
        if label == "thumbs_down":
            return self.thumbs_down_action
        return None


def run_rtsp_monitor(
    rtsp_url: str,
    frame_skip: int,
    min_confidence: float,
    *,
    display: bool = True,
    action_cooldown: float = 2.0,
    enable_home_assistant: bool = True,
) -> None:
    """Watch an RTSP stream, render predictions, and optionally trigger HA actions."""
    model, class_names, device = load_assets()

    thumbs_up_action: Optional[HAServiceAction] = None
    thumbs_down_action: Optional[HAServiceAction] = None

    if enable_home_assistant:
        light_entity = os.getenv("HA_THUMBS_LIGHT_ENTITY") or os.getenv("HA_LIGHT_ENTITY")
        default_up_payload: Dict[str, Any] = {}
        default_down_payload: Dict[str, Any] = {}
        if light_entity:
            default_up_payload["entity_id"] = light_entity
            default_down_payload["entity_id"] = light_entity

            brightness_env = os.getenv("THUMBS_UP_BRIGHTNESS_PCT")
            if brightness_env:
                try:
                    default_up_payload["brightness_pct"] = int(brightness_env)
                except ValueError:
                    logger.warning("Invalid THUMBS_UP_BRIGHTNESS_PCT value '%s'; ignoring.", brightness_env)

            color_env = os.getenv("THUMBS_UP_COLOR")
            if color_env:
                default_up_payload["color_name"] = color_env

        thumbs_up_action = load_action_from_env(
            "HA_THUMBS_UP",
            os.getenv("HA_THUMBS_UP_SERVICE", "light.turn_on" if light_entity else ""),
            default_up_payload if default_up_payload else None,
        )
        thumbs_down_action = load_action_from_env(
            "HA_THUMBS_DOWN",
            os.getenv("HA_THUMBS_DOWN_SERVICE", "light.turn_off" if light_entity else ""),
            default_down_payload if default_down_payload else None,
        )

        if thumbs_up_action is None and thumbs_down_action is None:
            logger.warning("No Home Assistant actions configured; running without automation.")
            enable_home_assistant = False

    bridge = (
        HomeAssistantGestureBridge(
            min_confidence=min_confidence,
            cooldown_seconds=action_cooldown,
            thumbs_up_action=thumbs_up_action,
            thumbs_down_action=thumbs_down_action,
        )
        if enable_home_assistant
        else None
    )

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open RTSP stream: {rtsp_url}")

    logger.info("Connected to RTSP stream. Press 'q' to exit.")
    frame_idx = 0
    last_prediction: Optional[Tuple[str, float, np.ndarray]] = None
    last_label: Optional[str] = None
    last_report_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from stream. Retrying...")
                time.sleep(0.5)
                continue

            frame_idx += 1
            should_classify = frame_idx % max(frame_skip, 1) == 0

            if should_classify:
                predicted_idx, confidence, probabilities = classify_frame(frame, model, device)
                label = class_names[predicted_idx]
                last_prediction = (label, confidence, probabilities)

                now = time.time()
                if confidence >= min_confidence and (label != last_label or now - last_report_time > 2.0):
                    breakdown = ", ".join(
                        f"{name}: {prob * 100:.1f}%"
                        for name, prob in zip(class_names, probabilities)
                    )
                    logger.info("Detected %s (%.1f%%) | %s", label, confidence * 100, breakdown)
                    last_label = label
                    last_report_time = now

                if bridge is not None:
                    bridge.handle(label, confidence)

            if display and last_prediction is not None:
                overlay_prediction(
                    frame,
                    class_names,
                    last_prediction[0],
                    last_prediction[1],
                    last_prediction[2],
                    min_confidence=min_confidence,
                )
                cv2.imshow("Thumbs Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif display:
                cv2.imshow("Thumbs Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


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
        help="Seconds to wait before repeating the same HA action (default 2).",
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
