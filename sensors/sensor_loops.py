from datetime import datetime
from typing import Optional
import logging
import time
import os

import app.database.database as db
from app.homeassistant.client import (
    get_boolean_state,
    get_numeric_state,
    set_light_state,
)

TEMPERATURE_SENSOR = os.getenv("HA_TEMPERATURE_SENSOR", "sensor.philio_sensor_temperature")
ILLUMINANCE_SENSOR = os.getenv("HA_ILLUMINANCE_SENSOR", "sensor.mco_air_monitor_illuminance")
MOTION_SENSOR = os.getenv("HA_MOTION_SENSOR", "binary_sensor.philio_sensor_motion")
CO2_SENSOR = os.getenv("HA_CO2_SENSOR", "sensor.mco_air_monitor_co2")
POLL_INTERVAL_SECONDS = int(os.getenv("HA_SENSOR_POLL_INTERVAL", "60"))
LIGHT_ENTITY = os.getenv("HA_LIGHT_ENTITY", "light.aeotec_led")


def read_sensor_loop():
    logging.info(
        "Starting Home Assistant polling loop (temperature=%s, illuminance=%s, motion=%s, co2=%s, interval=%ss).",
        TEMPERATURE_SENSOR,
        ILLUMINANCE_SENSOR,
        MOTION_SENSOR,
        CO2_SENSOR,
        POLL_INTERVAL_SECONDS,
    )

    while True:
        temp = get_numeric_state(TEMPERATURE_SENSOR)
        light = get_numeric_state(ILLUMINANCE_SENSOR)
        motion = get_boolean_state(MOTION_SENSOR)
        co2 = get_numeric_state(CO2_SENSOR)
        timestamp = datetime.now().isoformat()

        try:
            db.database_insert("environment", timestamp, temp, light, motion, co2)
            logging.info(
                "Sensor data recorded at %s | temperature=%s | illuminance=%s | motion=%s | co2=%s",
                timestamp,
                temp,
                light,
                motion,
                co2,
            )
        except Exception as exc:
            logging.exception("Failed to insert Home Assistant sensor snapshot: %s", exc)

        time.sleep(POLL_INTERVAL_SECONDS)


def automation_logic_loop():
    def _set_light(on: bool, *, brightness_pct: Optional[int] = None, color_name: Optional[str] = None) -> None:
        if not set_light_state(
            LIGHT_ENTITY,
            on,
            brightness_pct=brightness_pct,
            color_name=color_name,
        ):
            logging.error("Failed to set Home Assistant light '%s'.", LIGHT_ENTITY)

    last_state = None  # remember last light state to avoid redundant updates

    while True:
        try:
            latest_data = db.database_get_latest("environment")
            if not latest_data:
                logging.warning("No sensor data yet; waiting for data...")
                time.sleep(30)
                continue

            motion = bool(latest_data.get("motion", False))
            light = latest_data.get("illumination") or 0
            co2 = latest_data.get("co2")

            desired_on = False
            desired_brightness = 0
            desired_color = "warm white"

            # --- Motion & illumination logic ---
            if motion and light < 300:
                desired_on = True
                desired_brightness = 30
                desired_color = "warm white"
                logging.info("Motion detected and low light.")
            elif motion and light >= 300:
                desired_on = False
                logging.info("Motion detected but light sufficient.")
            else:
                desired_on = False
                logging.info("No motion detected.")

            # --- CO₂ override logic ---
            if co2 is None:
                logging.warning("CO₂ levels missing - no override applied.")
            elif co2 > 1000:
                desired_on = True
                desired_brightness = 60
                desired_color = "red"
                logging.warning("High CO₂ levels - overriding light to RED.")
            elif 800 <= co2 <= 1000:
                desired_on = True
                desired_brightness = 30
                desired_color = "yellow"
                logging.info("Moderate CO₂ levels - overriding light to YELLOW.")
            elif co2 < 800:
                logging.info("CO₂ levels normal.")

            # --- Apply only if state changed ---
            new_state = (desired_on, desired_brightness, desired_color)
            if new_state != last_state:
                _set_light(desired_on, brightness_pct=desired_brightness, color_name=desired_color)
                last_state = new_state
                logging.debug(f"Updated light state to {new_state}")
            else:
                logging.debug("Light state unchanged; skipping update.")

        except Exception as e:
            logging.exception(f"Error in automation logic loop: {e}")

        time.sleep(30)