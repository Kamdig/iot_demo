from sensors.sensor_loops import read_sensor_loop
from thumbs.thumbs import run_rtsp_monitor
import threading
import logging
import time
import os


# Launch sensor, automation, drone, and optional thumbs monitor threads.
def start_background_tasks():
    threading.Thread(target=read_sensor_loop, daemon=True).start()
    # Spin up the RTSP thumbs monitor only when enabled via configuration.
    if is_thumbs_monitor_enabled():
        threading.Thread(target=thumbs_monitor_loop, daemon=True, name="ThumbsMonitor").start()
    logging.info("Background sensor and automation threads started.")

def is_thumbs_monitor_enabled() -> bool:
    # Interpret the environment toggle allowing the thumbs monitor to run.
    value = os.getenv("THUMBS_MONITOR_ENABLED", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def thumbs_monitor_loop():
    # Continuously run the RTSP thumbs monitor with retry safeguards.
    url = os.getenv("THUMBS_RTSP_URL", "rtsp://iotworldcam:smart123@192.168.1.204/stream2")
    frame_skip = int(os.getenv("THUMBS_FRAME_SKIP", "2"))
    min_confidence = float(os.getenv("THUMBS_MIN_CONFIDENCE", "0.6"))
    cooldown = float(os.getenv("THUMBS_ACTION_COOLDOWN", "2.0"))
    display_window = os.getenv("THUMBS_DISPLAY_WINDOW", "1").strip().lower() in {"1", "true", "yes"}
    disable_ha = os.getenv("THUMBS_DISABLE_HA", "0").strip().lower() in {"1", "true", "yes"}

    logging.info("Starting thumbs monitor (display=%s, HA enabled=%s).", display_window, not disable_ha)

    # Keep restarting the monitor when it crashes to maintain coverage.
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
