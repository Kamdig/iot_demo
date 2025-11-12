from datetime import datetime
import logging
import time
import os

import app.database.database as db
from app.homeassistant.client import (
    get_boolean_state,
    get_numeric_state,
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