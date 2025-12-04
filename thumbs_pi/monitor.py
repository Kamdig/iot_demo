from __future__ import annotations
from app.homeassistant.client import HAServiceAction, HomeAssistantGestureBridge, load_action_from_env
from typing import Dict, Optional, Tuple
from .overlay import overlay_prediction
from .inference import classify_frame
import numpy as np
import logging
import time
import cv2
import os

logger = logging.getLogger(__name__)

def _build_home_assistant_bridge(
    *,
    min_confidence: float,
    action_cooldown: float,
    enable_home_assistant: bool,
) -> Optional[HomeAssistantGestureBridge]:
    if not enable_home_assistant:
        return None

    thumbs_up_action: Optional[HAServiceAction] = None
    thumbs_down_action: Optional[HAServiceAction] = None

    light_entity = os.getenv("HA_LIGHT_ENTITY")
    default_up_payload: Dict[str, object] = {}
    default_down_payload: Dict[str, object] = {}
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
        return None

    return HomeAssistantGestureBridge(
        min_confidence=min_confidence,
        cooldown_seconds=action_cooldown,
        thumbs_up_action=thumbs_up_action,
        thumbs_down_action=thumbs_down_action,
    )


def _detect_usb_webcam(max_device_index: int = 4) -> Optional[int]:
    """Return the first USB webcam index that can be opened, or None if none found."""
    for device_idx in range(max_device_index):
        cap = cv2.VideoCapture(device_idx)
        if not cap.isOpened():
            cap.release()
            continue

        ret, _ = cap.read()
        cap.release()
        if ret:
            return device_idx
    return None


def run_rtsp_monitor(
    rtsp_url: str,
    frame_skip: int,
    min_confidence: float,
    *,
    display: bool = True,
    action_cooldown: float = 2.0,
    enable_home_assistant: bool = True,
    bundle,
    class_names,
) -> None:
    """Watch an RTSP stream, render predictions, and optionally trigger Home Assistant actions."""
    bridge = _build_home_assistant_bridge(
        min_confidence=min_confidence,
        action_cooldown=action_cooldown,
        enable_home_assistant=enable_home_assistant,
    )

    numeric_index: Optional[int]
    try:
        numeric_index = int(rtsp_url)
    except (TypeError, ValueError):
        numeric_index = None

    capture_source = numeric_index if numeric_index is not None else rtsp_url
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        usb_index = _detect_usb_webcam()
        if usb_index is None:
            target_desc = f"RTSP stream '{rtsp_url}'" if numeric_index is None else f"camera index {numeric_index}"
            raise RuntimeError(f"Unable to open {target_desc}. No USB webcams detected either.")
        suggestion = (
            f"A USB webcam is available at index {usb_index}; try rerunning with '--rtsp-url {usb_index}'."
            if numeric_index is None
            else "A USB webcam responded during probing; verify it is not busy and you have permissions."
        )
        raise RuntimeError(
            f"Unable to open {('RTSP stream ' + rtsp_url) if numeric_index is None else f'camera index {numeric_index}'}. {suggestion}"
        )

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
                predicted_idx, confidence, probabilities = classify_frame(frame, bundle)
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

            if display:
                if last_prediction is not None:
                    overlay_prediction(
                        frame,
                        class_names=class_names,
                        label=last_prediction[0],
                        confidence=last_prediction[1],
                        probabilities=last_prediction[2],
                        min_confidence=min_confidence,
                    )
                cv2.imshow("Thumbs Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


__all__ = ["run_rtsp_monitor"]
