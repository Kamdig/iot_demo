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


# export LOG_LEVEL=DEBUG (to set logger to debug)
# export HOMEASSISTANT_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI0MmFiZWIxMGNmYmM0YWFhYjU2MWI5M2RmZGFhMTVjYSIsImlhdCI6MTc2MTU1NzEwNCwiZXhwIjoyMDc2OTE3MTA0fQ.0PH7iRiv7VHRtJUnPGDiyJTuiJaX0KrhkWnkH4RLr6c"
# export HOMEASSISTANT_BASE_URL="http://iotassistant.local:8123"
# export HA_TEMPERATURE_SENSOR="sensor.co2_monitor_air_quality_detector_air_temperature"
# export HA_ILLUMINANCE_SENSOR="sensor.multisensor_6_illuminance"
# export HA_MOTION_SENSOR="binary_sensor.multisensor_6_motion_detection"
# export HA_CO2_SENSOR="sensor.co2_monitor_air_quality_detector_carbon_dioxide_co2_level"
# export HA_LIGHT_ENTITY=light.bulb_6_multi_color
# export HA_THUMBS_LIGHT_ENTITY="light.bulb_6_multi_color"
