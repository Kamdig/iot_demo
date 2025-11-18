from __future__ import annotations

import logging
import os
import threading
from typing import Callable, List, Optional

from sensors.sensor_loops import read_sensor_loop


def _parse_env_flag(name: str) -> Optional[bool]:
    """Interpret common truthy/falsy strings from the environment."""
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    logging.warning("Ignoring invalid boolean value '%s' for %s.", value, name)
    return None


def _start_thread(target: Callable[[], None], *, name: str) -> threading.Thread:
    """Helper that consistently configures daemon threads and starts them."""
    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    return thread


def _thumbs_worker() -> None:
    """Background task that mirrors running ``python -m thumbs_pi``."""
    try:
        from thumbs_pi.monitor import run_rtsp_monitor
        from thumbs_pi.thumbs import parse_args
    except Exception:
        logging.exception("Thumbs monitor dependencies missing; set THUMBS_AUTOSTART=0 to disable.")
        return

    args = parse_args([])
    display_override = _parse_env_flag("THUMBS_DISPLAY_WINDOW")
    # Headless deployments default to no OpenCV window, but allow overrides for debugging.
    display = display_override if display_override is not None else False

    disable_override = _parse_env_flag("THUMBS_DISABLE_HA")
    disable_ha = disable_override if disable_override is not None else args.disable_ha

    logging.info(
        "Starting thumbs monitor thread (source=%s, frame_skip=%s, min_confidence=%s, display=%s, ha=%s).",
        args.rtsp_url,
        args.frame_skip,
        args.min_confidence,
        display,
        "enabled" if not disable_ha else "disabled",
    )
    try:
        run_rtsp_monitor(
            rtsp_url=args.rtsp_url,
            frame_skip=args.frame_skip,
            min_confidence=args.min_confidence,
            display=display,
            action_cooldown=args.action_cooldown,
            enable_home_assistant=not disable_ha,
        )
    except Exception:
        logging.exception("Thumbs monitor thread terminated due to an unexpected error.")


def _should_start_thumbs() -> bool:
    """Return whether the thumbs monitor thread should be auto-started."""
    override = _parse_env_flag("THUMBS_AUTOSTART")
    if override is not None:
        return override
    return True


# Launch sensor, automation, drone, and optional thumbs monitor threads.
def start_background_tasks() -> List[threading.Thread]:
    """Spin up all long-lived background services and return the thread handles."""
    threads: List[threading.Thread] = []
    threads.append(_start_thread(read_sensor_loop, name="ha-sensor-poll"))
    logging.info("Home Assistant sensor polling thread started.")

    if _should_start_thumbs():
        threads.append(_start_thread(_thumbs_worker, name="thumbs-monitor"))
    else:
        logging.info("THUMBS_AUTOSTART disabled; skipping thumbs monitor thread.")
    return threads
