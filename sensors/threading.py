from sensors.sensor_loops import read_sensor_loop, automation_logic_loop
from app.mqtt.mqtt_client import drone_trigger, fly_sequence
from thumbs_ai.thumbs import run_rtsp_monitor

import threading
import logging
import time
import os

def start_background_tasks():
    threading.Thread(target=read_sensor_loop, daemon=True).start()
    threading.Thread(target=automation_logic_loop, daemon=True).start()
    threading.Thread(target=drone_listener_loop, daemon=True).start()
    if is_thumbs_monitor_enabled():
        threading.Thread(target=thumbs_monitor_loop, daemon=True, name="ThumbsMonitor").start()
    logging.info("Background sensor and automation threads started.")

def drone_listener_loop():
    global drone_trigger
    while True:
        try:
            if drone_trigger:
                logging.info("Drone trigger detected, starting flight sequence...")
                fly_sequence()
                drone_trigger = False
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error in drone listener loop: {e}")
        time.sleep(1)


def is_thumbs_monitor_enabled() -> bool:
    value = os.getenv("THUMBS_MONITOR_ENABLED", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def thumbs_monitor_loop():
    url = os.getenv("THUMBS_RTSP_URL", "rtsp://iotworldcam:smart123@10.136.171.24/stream2")
    frame_skip = int(os.getenv("THUMBS_FRAME_SKIP", "2"))
    min_confidence = float(os.getenv("THUMBS_MIN_CONFIDENCE", "0.6"))
    cooldown = float(os.getenv("THUMBS_ACTION_COOLDOWN", "2.0"))
    display_window = os.getenv("THUMBS_DISPLAY_WINDOW", "1").strip().lower() in {"1", "true", "yes"}
    disable_ha = os.getenv("THUMBS_DISABLE_HA", "0").strip().lower() in {"1", "true", "yes"}

    logging.info("Starting thumbs monitor (display=%s, HA enabled=%s).", display_window, not disable_ha)

    while True:
        try:
            run_rtsp_monitor(
                rtsp_url=url,
                frame_skip=frame_skip,
                min_confidence=min_confidence,
                display=display_window,
                action_cooldown=cooldown,
                enable_home_assistant=not disable_ha,
            )
            logging.info("Thumbs monitor loop exited normally.")
            break
        except Exception as exc:
            logging.exception("Thumbs monitor encountered an error: %s. Retrying in 5 seconds.", exc)
            time.sleep(5)
