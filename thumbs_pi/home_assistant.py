from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Thin wrapper around the Home Assistant REST API."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        )

    def call_service(self, domain: str, service: str, data: Dict[str, Any]) -> bool:
        url = f"{self.base_url}/api/services/{domain}/{service}"
        try:
            response = self._session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError as exc:
            logger.error(
                "HTTP error calling Home Assistant service %s/%s with payload %s: %s",
                domain,
                service,
                data,
                exc,
            )
        except requests.exceptions.Timeout:
            logger.error(
                "Timeout calling Home Assistant service %s/%s after %s seconds.",
                domain,
                service,
                self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            logger.error(
                "Connection error calling Home Assistant service %s/%s: %s",
                domain,
                service,
                exc,
            )
        return False


def _build_client_from_env() -> Optional[HomeAssistantClient]:
    token = os.getenv("HOMEASSISTANT_TOKEN")
    if not token:
        logger.warning("HOMEASSISTANT_TOKEN not set; Home Assistant integration disabled.")
        return None

    base_url = os.getenv("HOMEASSISTANT_BASE_URL", "http://homeassistant.local:8123")
    timeout_value = 10.0
    timeout_env = os.getenv("HOMEASSISTANT_TIMEOUT")
    if timeout_env:
        try:
            timeout_value = float(timeout_env)
        except ValueError:
            logger.warning("Invalid HOMEASSISTANT_TIMEOUT value '%s'; falling back to 10 seconds.", timeout_env)

    return HomeAssistantClient(base_url=base_url, token=token, timeout=timeout_value)


ClientFactory = Callable[[], Optional[HomeAssistantClient]]

_client_factory: Optional[ClientFactory] = None
_client_cache: Optional[HomeAssistantClient] = None


def set_client_factory(factory: Optional[ClientFactory]) -> None:
    """Override Home Assistant client creation (primarily for testing)."""
    global _client_factory, _client_cache
    _client_factory = factory
    _client_cache = None


def get_client() -> Optional[HomeAssistantClient]:
    """Return a cached Home Assistant client or build one from environment."""
    global _client_cache
    if _client_cache is not None:
        return _client_cache

    factory = _client_factory or _build_client_from_env
    client = factory()
    if client is None:
        return None

    _client_cache = client
    return client


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


def parse_service_string(service: str) -> tuple[str, str]:
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
    """Build an HAServiceAction from environment variables, falling back to defaults."""
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


__all__ = [
    "HAServiceAction",
    "HomeAssistantClient",
    "HomeAssistantGestureBridge",
    "get_client",
    "load_action_from_env",
    "parse_service_string",
    "set_client_factory",
]
