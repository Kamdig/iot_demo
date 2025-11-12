from __future__ import annotations

from app.homeassistant.client import get_client
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import logging
import time
import json
import os


logger = logging.getLogger(__name__)


@dataclass
class HAServiceAction:
    """Wrapper around a Home Assistant service invocation."""

    domain: str
    service: str
    payload: Dict[str, Any]

    def execute(self) -> bool:
        # Attempt to call the configured Home Assistant service.
        client = get_client()
        # Abort if the shared client is unavailable.
        if client is None:
            logger.error(
                "Home Assistant client unavailable; cannot execute %s.%s with payload %s",
                self.domain,
                self.service,
                self.payload,
            )
            return False

        success = client.call_service(self.domain, self.service, self.payload)
        # Log success/failure so operators can diagnose automation issues.
        if success:
            logger.info("Home Assistant service %s.%s executed with %s", self.domain, self.service, self.payload)
        else:
            logger.error("Home Assistant service %s.%s failed for payload %s", self.domain, self.service, self.payload)
        return success


# Validate and split a Home Assistant service string into domain/service.
# Validate and split a Home Assistant service string into domain/service.
def parse_service_string(service: str) -> tuple[str, str]:
    """Split a Home Assistant service string of the form 'domain.service'."""
    # Require the canonical "domain.service" format.
    if "." not in service:
        raise ValueError(f"Service '{service}' must be in the format 'domain.service'.")
    domain, service_name = service.split(".", 1)
    # Ensure both halves of the string contain characters.
    if not domain or not service_name:
        raise ValueError(f"Service '{service}' must be in the format 'domain.service'.")
    return domain, service_name


# Build an HAServiceAction by reading prefixed environment variables.
# Build an HAServiceAction by reading prefixed environment variables.
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
    # Skip configuration when no service value exists.
    if not service_value:
        logger.debug("%s not configured; skipping service action.", service_key)
        return None

    payload: Dict[str, Any] = {}
    # Seed the payload with defaults such as entity_id.
    if default_payload:
        payload.update(default_payload)

    payload_override = os.getenv(payload_key)
    # Merge JSON payload overrides from the environment.
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

    # Evaluate a detection result and fire actions when appropriate.
    def handle(self, label: str, confidence: float) -> None:
        # Ignore detections that do not meet the confidence threshold.
        if confidence < self.min_confidence:
            return

        action = self._select_action(label)
        # Skip labels that have no configured automation.
        if action is None:
            return

        now = time.time()
        last_fired = self._last_triggered.get(label, 0.0)
        # Respect the cooldown between repeated gesture triggers.
        if now - last_fired < self.cooldown_seconds:
            return

        # Only update cooldown tracking when the service call succeeds.
        if action.execute():
            self._last_triggered[label] = now

    # Choose which HA action applies for a given label.
    def _select_action(self, label: str) -> Optional[HAServiceAction]:
        # Route the label to the configured thumbs-up action.
        if label == "thumbs_up":
            return self.thumbs_up_action
        # Route the label to the configured thumbs-down action.
        if label == "thumbs_down":
            return self.thumbs_down_action
        return None


__all__ = ["HAServiceAction", "HomeAssistantGestureBridge", "load_action_from_env"]
