from sensors.sensor_loops import read_sensor_loop, automation_logic_loop
from app.mqtt.mqtt_client import drone_trigger, fly_sequence
import threading
import logging
import time

def start_background_tasks():
    threading.Thread(target=read_sensor_loop, daemon=True).start()
    threading.Thread(target=automation_logic_loop, daemon=True).start()
    threading.Thread(target=drone_listener_loop, daemon=True).start()
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