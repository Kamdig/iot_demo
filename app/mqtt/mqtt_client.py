import paho.mqtt.client as mqtt
import logging

drone_trigger = False

BROKER = "iotassistant.local"
PORT = 1883
USERNAME = "mqtt_user"
PASSWORD = "smart123"

client = mqtt.Client(client_id="flask_app", protocol=mqtt.MQTTv311)
client.username_pw_set(USERNAME, PASSWORD)

def on_connect(client, userdata, flags, rc):
    logging.info(f"Connected to MQTT Broker with result code {rc}")
    client.subscribe("booth/command/#")
    logging.info("Subscribed to topic 'booth/command/#'")

def on_message(client, userdata, msg):
    payload = msg.payload.decode().lower()
    logging.info(f"Received message on {msg.topic}: {payload}")


client.on_connect = on_connect
client.on_message = on_message

def initialize_mqtt():
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        logging.info("MQTT client initialized and loop started.")
    except Exception as e:
        logging.error(f"Failed to connect to MQTT Broker: {e}")

def publish_message(topic, payload):
    client.publish(topic, payload)
    logging.info(f"Published message to {topic}: {payload}")