from __future__ import annotations

from .home_assistant import HAServiceAction, HomeAssistantGestureBridge, load_action_from_env
from typing import Dict, Optional, Sequence, Tuple
from .overlay import overlay_prediction
from .inference import classify_frame
from .assets import load_assets
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
    # Configure the optional Home Assistant bridge as dictated by env vars.
    # Skip building the bridge entirely when HA integration is disabled.
    if not enable_home_assistant:
        return None

    thumbs_up_action: Optional[HAServiceAction] = None
    thumbs_down_action: Optional[HAServiceAction] = None

    light_entity = os.getenv("HA_THUMBS_LIGHT_ENTITY") or os.getenv("HA_LIGHT_ENTITY")
    default_up_payload: Dict[str, object] = {}
    default_down_payload: Dict[str, object] = {}
    # Pre-fill payload defaults when a light entity is provided.
    if light_entity:
        default_up_payload["entity_id"] = light_entity
        default_down_payload["entity_id"] = light_entity

        brightness_env = os.getenv("THUMBS_UP_BRIGHTNESS_PCT")
        # Accept brightness overrides from the environment.
        if brightness_env:
            try:
                default_up_payload["brightness_pct"] = int(brightness_env)
            except ValueError:
                logger.warning("Invalid THUMBS_UP_BRIGHTNESS_PCT value '%s'; ignoring.", brightness_env)

        color_env = os.getenv("THUMBS_UP_COLOR")
        # Allow an optional color override for thumbs up events.
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

    # Skip creating the bridge when neither action is configured.
    if thumbs_up_action is None and thumbs_down_action is None:
        logger.warning("No Home Assistant actions configured; running without automation.")
        return None

    return HomeAssistantGestureBridge(
        min_confidence=min_confidence,
        cooldown_seconds=action_cooldown,
        thumbs_up_action=thumbs_up_action,
        thumbs_down_action=thumbs_down_action,
    )


# Watch an RTSP stream, overlay predictions, and optionally trigger HA actions.
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
    bridge = _build_home_assistant_bridge(
        min_confidence=min_confidence,
        action_cooldown=action_cooldown,
        enable_home_assistant=enable_home_assistant,
    )

    cap = cv2.VideoCapture(rtsp_url)
    # Fail immediately when the RTSP stream cannot be opened.
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open RTSP stream: {rtsp_url}")

    logger.info("Connected to RTSP stream. Press 'q' to exit.")
    frame_idx = 0
    last_prediction: Optional[Tuple[str, float, np.ndarray]] = None
    last_label: Optional[str] = None
    last_report_time = 0.0

    try:
        # Process frames until the user quits or an exception occurs.
        while True:
            ret, frame = cap.read()
            # Wait briefly and retry if the stream temporarily fails.
            if not ret:
                logger.warning("Failed to read frame from stream. Retrying...")
                time.sleep(0.5)
                continue

            frame_idx += 1
            should_classify = frame_idx % max(frame_skip, 1) == 0

            # Only run inference on frames selected by the skip interval.
            if should_classify:
                predicted_idx, confidence, probabilities = classify_frame(frame, model, device)
                label = class_names[predicted_idx]
                last_prediction = (label, confidence, probabilities)

                now = time.time()
                # Log detections that meet the confidence threshold and cooldown rules.
                if confidence >= min_confidence and (label != last_label or now - last_report_time > 2.0):
                    breakdown = ", ".join(
                        f"{name}: {prob * 100:.1f}%"
                        for name, prob in zip(class_names, probabilities)
                    )
                    logger.info("Detected %s (%.1f%%) | %s", label, confidence * 100, breakdown)
                    last_label = label
                    last_report_time = now

                # Trigger the Home Assistant action bridge when configured.
                if bridge is not None:
                    bridge.handle(label, confidence)

            # Render the annotated frame only when windowed display is enabled.
            if display:
                # Only draw overlays after at least one prediction has been made.
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
                # Allow users to exit the preview window with "q".
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        # Clean up OpenCV windows only when rendering was enabled.
        if display:
            cv2.destroyAllWindows()


__all__ = ["run_rtsp_monitor"]
