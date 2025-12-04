"""
Compatibility shim: thumbs_pi now re-uses the shared Home Assistant client in app.homeassistant.client.
"""
from app.homeassistant.client import (  # noqa: F401
    HAServiceAction,
    HomeAssistantClient,
    HomeAssistantGestureBridge,
    get_client,
    load_action_from_env,
    parse_service_string,
    set_client_factory,
)

__all__ = [
    "HAServiceAction",
    "HomeAssistantClient",
    "HomeAssistantGestureBridge",
    "get_client",
    "load_action_from_env",
    "parse_service_string",
    "set_client_factory",
]
