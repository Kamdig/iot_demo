from app.database.database import initialize_database, database_insert
from app.mqtt.mqtt_client import initialize_mqtt, publish_message
from sensors.threading import start_background_tasks
from app.logger.logger import initialize_logger
from app.app import create_app
from datetime import datetime
from time import sleep
import logging
import random

if __name__ == "__main__":
    initialize_logger()
    initialize_database()
    logging.info("Starting main application...")
    for i in range(10):
        database_insert("environment", datetime.now(), random.randint(0, 50), random.randint(0, 1000), random.choice([True, False]), random.randint(0, 2000)) #table, timestamp, temp, light, motion, co2
        sleep(1)
    initialize_mqtt()
    publish_message("home/automation/status", "Application started")
    #start_background_tasks()
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)