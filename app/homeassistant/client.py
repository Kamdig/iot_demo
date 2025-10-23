from typing import Any, Dict, Optional
import logging
import os

import requests

_cached_client: "Optional[HomeAssistantClient]" = None


class HomeAssistantClient:
    """Thin wrapper around the Home Assistant REST API for reading entity state."""

    def __init__(
        self,
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

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def get_entity_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Return the raw state payload for the provided entity ID."""
        url = f"{self.base_url}/api/states/{entity_id}"
        try:
            response = requests.get(url, headers=self._headers(), timeout=self.timeout)
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
            response = requests.post(url, headers=self._headers(), json=data, timeout=self.timeout)
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

    def set_light_state(
        self,
        entity_id: str,
        on: bool,
        *,
        brightness_pct: Optional[int] = None,
        color_name: Optional[str] = None,
    ) -> bool:
        payload: Dict[str, Any] = {"entity_id": entity_id}

        if brightness_pct is not None:
            payload["brightness_pct"] = max(0, min(100, int(brightness_pct)))
        if color_name:
            payload["color_name"] = color_name

        if on:
            return self.call_service("light", "turn_on", payload)
        return self.call_service("light", "turn_off", {"entity_id": entity_id})


def get_client() -> Optional[HomeAssistantClient]:
    """Return a cached HomeAssistantClient instance, if configuration permits."""
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    try:
        _cached_client = HomeAssistantClient()
    except ValueError as exc:
        logging.error("Home Assistant client configuration error: %s", exc)
        return None
    return _cached_client


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


def set_light_state(
    entity_id: str,
    on: bool,
    *,
    brightness_pct: Optional[int] = None,
    color_name: Optional[str] = None,
) -> bool:
    """Set a Home Assistant light entity on/off with optional brightness and color."""
    client = get_client()
    if client is None:
        logging.error("Cannot control light '%s' because Home Assistant client failed to initialize.", entity_id)
        return False

    return client.set_light_state(
        entity_id,
        on,
        brightness_pct=brightness_pct,
        color_name=color_name,
    )
