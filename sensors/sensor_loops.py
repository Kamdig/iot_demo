from app.zwave.zwave import zwave_read, zwave_send_command
import app.database.database as db
from datetime import datetime
import logging
import time

def read_sensor_loop():
    while True:
        temp = zwave_read("philio_sensor", "Temperature")
        light = zwave_read("mco_air_monitor", "Illuminance")
        motion = zwave_read("philio_sensor", "Motion")
        co2 = zwave_read("mco_air_monitor", "CO2")
        timestamp = datetime.now().isoformat()

        db.database_insert("environment", timestamp, temp, light, motion, co2)
        logging.info(f"Sensor data recorded at {timestamp}")
        logging.info(f"Temperature: {temp}°C, Light: {light}, Motion: {motion}, CO2: {co2}")
        time.sleep(60)

def automation_logic_loop():
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

            if motion and light < 300:
                zwave_send_command("aeotec_led", "on", brightness=30, color="warm white")
                logging.info("Motion detected and low light - LED turned ON")
            elif not motion:
                zwave_send_command("aeotec_led", "off")
                logging.info("No motion detected - LED turned OFF")
            elif motion and light >= 300:
                zwave_send_command("aeotec_led", "off")
                logging.info("Motion detected but sufficient light - LED turned OFF")
            else:
                logging.info("No action taken based on motion and light levels")
            
            if co2 is not None and co2 > 1000:
                zwave_send_command("aeotec_led", "on", brightness=60, color="red")
                logging.warning("High CO2 levels detected - LED turned RED")
            elif co2 is not None and 800 <= co2 <= 1000:
                zwave_send_command("aeotec_led", "on", brightness=30, color="yellow")
                logging.info("Moderate CO2 levels - LED turned YELLOW")
            elif co2 is not None and co2 < 800:
                zwave_send_command("aeotec_led", "off")
                logging.info("CO2 levels normal - LED turned OFF")
            else:
                logging.info("CO2 levels missing - No action taken")

        except Exception as e:
            logging.exception(f"Error in automation logic loop: {e}")
            
        time.sleep(30)