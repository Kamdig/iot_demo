from cflib.crazyflie import Crazyflie
import paho.mqtt.client as mqtt
import cflib.crtp
import logging
import time

drone_trigger = False

BROKER = "iotassistant.local"
PORT = 1883
USERNAME = "mqtt_user"
PASSWORD = "smart123"

CRAZYFLIE_URI = "radio://0/80/2M"

cflib.crtp.init_drivers(enable_debug_driver=False)
crazyflie = Crazyflie(rw_cache="./cache")

client = mqtt.Client(client_id="flask_app", protocol=mqtt.MQTTv311)
client.username_pw_set(USERNAME, PASSWORD)

def fly_sequence():
    try:
        logging.info("Connecting to Crazyflie...")
        crazyflie.open_link(CRAZYFLIE_URI)
        time.sleep(1)

        logging.info("Taking off...")
        crazyflie.commander.send.setpoint(0, 0, 0, 40000)
        time.sleep(0.2)

        logging.info("Hovering...")
        for _ in range(10):
            crazyflie.commander.send.setpoint(0, 0, 0, 35000)
            time.sleep(0.1)

        logging.info("Landing...")
        for _ in range(10):
            crazyflie.commander.send.setpoint(0, 0, 0, 20000)
            time.sleep(1)
        
        crazyflie.commander.send_stop_setpoint()
        crazyflie.close_link()
        logging.info("Flight sequence completed.")
    except Exception as e:
        logging.error(f"Error during flight sequence: {e}")

def on_connect(client, userdata, flags, rc):
    logging.info(f"Connected to MQTT Broker with result code {rc}")
    client.subscribe("booth/command/#")
    logging.info("Subscribed to topic 'booth/command/#'")

def on_message(client, userdata, msg):
    payload = msg.payload.decode().lower()
    logging.info(f"Received message on {msg.topic}: {payload}")

    if msg.topic == "booth/command/drone":
        if payload == "on":
            logging.info("Drone trigger received (queued for flight)...")
            global drone_trigger
            drone_trigger = True


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