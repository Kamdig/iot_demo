from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field
import requests
import logging
import json
import time
import os

ClientFactory = Callable[[], Optional["HomeAssistantClient"]]

_client_factory: Optional[ClientFactory] = None
_client_cache: Optional["HomeAssistantClient"] = None


class HomeAssistantClient:
    """Thin wrapper around the Home Assistant REST API for state + service calls."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        default_base = os.getenv("HOMEASSISTANT_BASE_URL", "http://homeassistant.local:8123")
        self.base_url = (base_url or default_base).rstrip("/")
        self.token = token or os.getenv("HOMEASSISTANT_TOKEN")
        if not self.token:
            raise ValueError("HOMEASSISTANT_TOKEN not set; create a long-lived access token in Home Assistant.")

        env_timeout = os.getenv("HOMEASSISTANT_TIMEOUT")
        if timeout is not None:
            self.timeout = timeout
        elif env_timeout:
            try:
                self.timeout = float(env_timeout)
            except ValueError:
                logging.warning("Invalid HOMEASSISTANT_TIMEOUT value '%s'; falling back to 10 seconds.", env_timeout)
                self.timeout = 10.0
        else:
            self.timeout = 10.0

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
        )

    def get_entity_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Return the raw state payload for the provided entity ID."""
        url = f"{self.base_url}/api/states/{entity_id}"
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logging.warning("Home Assistant entity '%s' not found (404).", entity_id)
            else:
                logging.error("HTTP error retrieving entity '%s': %s", entity_id, exc)
        except requests.exceptions.Timeout:
            logging.error("Request to Home Assistant for '%s' timed out after %s seconds.", entity_id, self.timeout)
        except requests.exceptions.RequestException as exc:
            logging.error("Connection error retrieving entity '%s': %s", entity_id, exc)
        except ValueError as exc:
            logging.error("Failed to parse JSON response for '%s': %s", entity_id, exc)
        return None

    def call_service(self, domain: str, service: str, data: Dict[str, Any]) -> bool:
        """Invoke a Home Assistant service and return True on success."""
        url = f"{self.base_url}/api/services/{domain}/{service}"
        try:
            response = self._session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError as exc:
            logging.error(
                "HTTP error calling Home Assistant service %s/%s with payload %s: %s",
                domain,
                service,
                data,
                exc,
            )
        except requests.exceptions.Timeout:
            logging.error(
                "Timeout calling Home Assistant service %s/%s after %s seconds.",
                domain,
                service,
                self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            logging.error(
                "Connection error calling Home Assistant service %s/%s: %s",
                domain,
                service,
                exc,
            )
        return False


def _build_client_from_env() -> Optional[HomeAssistantClient]:
    token = os.getenv("HOMEASSISTANT_TOKEN")
    if not token:
        logging.warning("HOMEASSISTANT_TOKEN not set; Home Assistant integration disabled.")
        return None

    base_url = os.getenv("HOMEASSISTANT_BASE_URL", "http://homeassistant.local:8123")
    timeout_value = 10.0
    timeout_env = os.getenv("HOMEASSISTANT_TIMEOUT")
    if timeout_env:
        try:
            timeout_value = float(timeout_env)
        except ValueError:
            logging.warning("Invalid HOMEASSISTANT_TIMEOUT value '%s'; falling back to 10 seconds.", timeout_env)

    return HomeAssistantClient(base_url=base_url, token=token, timeout=timeout_value)


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
    try:
        client = factory()
    except ValueError as exc:
        logging.error("Home Assistant client configuration error: %s", exc)
        return None

    if client is None:
        return None

    _client_cache = client
    return _client_cache


def get_numeric_state(entity_id: str) -> Optional[float]:
    """Retrieve a numeric sensor value from Home Assistant."""
    client = get_client()
    if client is None:
        return None

    payload = client.get_entity_state(entity_id)
    if not payload:
        return None

    raw_state = payload.get("state")
    if raw_state in (None, "", "unknown", "unavailable"):
        logging.debug("Entity '%s' returned non-numeric state '%s'.", entity_id, raw_state)
        return None

    try:
        return float(raw_state)
    except (TypeError, ValueError):
        logging.warning("Could not convert Home Assistant state '%s' for '%s' to float.", raw_state, entity_id)
        return None


def get_boolean_state(entity_id: str) -> Optional[bool]:
    """Retrieve a binary sensor value from Home Assistant."""
    client = get_client()
    if client is None:
        return None

    payload = client.get_entity_state(entity_id)
    if not payload:
        return None

    raw_state = payload.get("state")
    if raw_state is None:
        logging.debug("Entity '%s' returned no state.", entity_id)
        return None

    normalized = str(raw_state).strip().lower()
    if normalized in {"on", "true", "1", "open", "detected"}:
        return True
    if normalized in {"off", "false", "0", "closed", "clear"}:
        return False

    logging.warning("Unrecognized boolean state '%s' for entity '%s'.", raw_state, entity_id)
    return None


@dataclass
class HAServiceAction:
    """Wrapper around a Home Assistant service invocation."""

    domain: str
    service: str
    payload: Dict[str, Any]

    def execute(self) -> bool:
        client = get_client()
        if client is None:
            logging.error(
                "Home Assistant client unavailable; cannot execute %s.%s with payload %s",
                self.domain,
                self.service,
                self.payload,
            )
            return False

        success = client.call_service(self.domain, self.service, self.payload)
        if success:
            logging.info("Home Assistant service %s.%s executed with %s", self.domain, self.service, self.payload)
        else:
            logging.error("Home Assistant service %s.%s failed for payload %s", self.domain, self.service, self.payload)
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
        logging.debug("%s not configured; skipping service action.", service_key)
        return None

    payload: Dict[str, Any] = {}
    if default_payload:
        payload.update(default_payload)

    payload_override = os.getenv(payload_key)
    if payload_override:
        try:
            payload.update(json.loads(payload_override))
        except json.JSONDecodeError as exc:
            logging.error("Invalid JSON for %s: %s", payload_key, exc)

    try:
        domain, service_name = parse_service_string(service_value)
    except ValueError as exc:
        logging.error("Skipping Home Assistant action for %s: %s", service_key, exc)
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
    "get_numeric_state",
    "get_boolean_state",
    "load_action_from_env",
    "parse_service_string",
    "set_client_factory",
]
