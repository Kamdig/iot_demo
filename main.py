from app.mqtt.mqtt_client import initialize_mqtt, publish_message
from app.database.database import initialize_database
from sensors.threading import start_background_tasks
from app.logger.logger import initialize_logger
from app.app import create_app
import logging

# Entry point: wire up infrastructure and start the Flask app server.
if __name__ == "__main__":
    # Bring up logging first so later steps report rich output to both console and file.
    initialize_logger()
    # Ensure the SQLite database schema exists before any background threads touch it.
    initialize_database()
    logging.info("Starting main application...")
    # Connect to MQTT as early as possible so downstream services can publish events.
    initialize_mqtt()
    publish_message("home/automation/status", "Application started")
    # Kick off all background loops (sensor polling + thumbs monitor).
    start_background_tasks()
    # Create and serve the Flask dashboard plus REST API.
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)