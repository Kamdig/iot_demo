import paho.mqtt.client as mqtt
import logging
import os

def _build_mqtt_client() -> mqtt.Client:
    """Configure the MQTT client with env-provided credentials (plaintext)."""
    global _BROKER, _PORT

    _BROKER = os.getenv("MQTT_BROKER")
    _PORT = int(os.getenv("MQTT_PORT", "1883"))
    username = os.getenv("MQTT_USERNAME")
    password = os.getenv("MQTT_PASSWORD")
    client_id = os.getenv("MQTT_CLIENT_ID", "flask_app")

    client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)

    if username and password:
        client.username_pw_set(username, password)
    else:
        logging.warning("MQTT credentials not set; broker may reject unauthenticated connections.")

    return client


client = _build_mqtt_client()

def on_connect(client, userdata, flags, rc):
    logging.info(f"Connected to MQTT Broker with result code {rc}")
    client.subscribe("booth/command/#")
    logging.info("Subscribed to topic 'booth/command/#'")

def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", errors="replace").lower()
    logging.info(f"Received message on {msg.topic}: {payload}")


client.on_connect = on_connect
client.on_message = on_message

def initialize_mqtt():
    try:
        client.connect(_BROKER, _PORT, 60)
        client.loop_start()
        logging.info("MQTT client initialized and loop started.")
    except Exception as e:
        logging.error(f"Failed to connect to MQTT Broker: {e}")

def publish_message(topic, payload):
    client.publish(topic, payload)
    logging.info(f"Published message to {topic}: {payload}")
