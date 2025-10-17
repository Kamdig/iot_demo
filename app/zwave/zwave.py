from openzwavemqtt import OZWManager
import logging

zwave_manager = None
devices = {}

def initialize_zwave_controller():
    global zwave_manager, devices

    zwave_manager = OZWManager(
        mqtt_broker_host="localhost",
        mqtt_broker_port=1883,
        topic_prefix="OpenZWave"
    )

    devices = {
        "philio_sensor": 1,
        "mco_air_monitor": 2,
        "aeotec_led": 3,
        "nanomote": 4
    }

    zwave_manager.connect()
    logging.info("Z-Wave controller initialized and connected to MQTT broker.")

    for name, node_id in devices.items():
        logging.info(f"Registered device '{name}' with Node ID {node_id}")
    
    logging.info("Z-Wave devices initialized.")

def zwave_read(device_name, value_id):
    try:
        node_id = devices.get(device_name)
        if not node_id:
            raise ValueError(f"Device '{device_name}' not found.")

        node = zwave_manager.get_node(node_id)
        if not node:
            logging.warning(f"Node ID {node_id} for device '{device_name}' not found.")
            return None
        
        for val in node.values.values():
            if val.label.lower() == value_id.lower():
                logging.info(f"Read value '{val.label}' from device '{device_name}': {val.value}")
                return val.value
        
        logging.warning(f"Value ID '{value_id}' not found for device '{device_name}'.")
        return None
    
    except Exception as e:
        logging.error(f"Error reading from device '{device_name}': {e}")
        return None
    
def zwave_send_command(device_name, command, **kwargs):
    try:
        node_id = devices.get(device_name)
        if not node_id:
            raise ValueError(f"Unknown device '{device_name}'.")
        
        node = zwave_manager.get_node(node_id)
        if not node:
            logging.warning(f"Node ID {node_id} for device '{device_name}' not found.")
            return False
        
        if command == "on":
            node.set_value("Switch", True)
            logging.info(f"Sent 'ON' command to device '{device_name}'.")
        elif command == "off":
            node.set_value("Switch", False)
            logging.info(f"Sent 'OFF' command to device '{device_name}'.")
        elif command == "color":
            color = kwargs.get("color", "white")
            node.set_value("Color", color)
            logging.info(f"Set color of device '{device_name}' to '{color}'.")
        elif command == "brightness":
            level = kwargs.get("level", 100)
            node.set_value("Brightness", level)
            logging.info(f"Set brightness of device '{device_name}' to '{level}'.")
        else:
            logging.warning(f"Unknown command '{command}' for device '{device_name}'.")
            return False
    except Exception as e:
        logging.error(f"Error sending command to device '{device_name}': {e}")
