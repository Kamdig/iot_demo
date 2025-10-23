from app.database.database import initialize_database
from app.mqtt.mqtt_client import initialize_mqtt, publish_message
from sensors.threading import start_background_tasks
from app.logger.logger import initialize_logger
from app.app import create_app
import logging

if __name__ == "__main__":
    initialize_logger()
    initialize_database()
    logging.info("Starting main application...")
    initialize_mqtt()
    publish_message("home/automation/status", "Application started")
    start_background_tasks()
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)


# export LOG_LEVEL=DEBUG TO SET DEBUG LEVEL
# export HOMEASSISTANT_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1Nzc0ZWRlODhmYjk0MGNkYjM1MjNjMTgzOGU2YTEwNyIsImlhdCI6MTc2MDY5MTM2MCwiZXhwIjoyMDc2MDUxMzYwfQ.dL4jt0fDsUQ4-eNlj3G9Cv-EGOlOt0bZ8lNLaRKz5J4"
# export HOMEASSISTANT_BASE_URL="http://iotassistant.local:8123"
# export HA_TEMPERATURE_SENSOR="sensor.co2_monitor_air_quality_detector_air_temperature"
# export HA_ILLUMINANCE_SENSOR="sensor.multisensor_6_illuminance"
# export HA_MOTION_SENSOR="binary_sensor.multisensor_6_motion_detection"
# export HA_CO2_SENSOR="sensor.co2_monitor_air_quality_detector_carbon_dioxide_co2_level"
# export HA_LIGHT_ENTITY=light.bulb_6_multi_color