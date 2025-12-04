# IoT Gesture Hub – Thumbs, Lights & Drones

This project turns a simple webcam into a **gesture-driven control hub**:

- A **Camera over RTSP** looks at you.
- Two **Raspberry Pi's**:
  - Runs a **TFLite gesture classifier** (`thumbs_up`, `thumbs_down`, `neutral`).
  - Serves a **Flask dashboard** with:
    - `/video` – MJPEG stream with live overlayed predictions.
    - `/api/data` – recent environment sensor values from the database.
    - `/api/logs` – tail of log files for debugging.
  - Sends **webhooks to Home Assistant** when the detected gesture changes.
  - (Optional) Controls a **Crazyflie 2.0 drone** to perform a small “hop” on `thumbs_up`.
- **Home Assistant** reacts to gestures:
  - `thumbs_up` → turn on a specific light with configured color & brightness.
  - `thumbs_down` → turn off that light.
  - `neutral` → optionally turn off the light / treat as idle.

The **training pipeline** runs on a PC/Mac with full TensorFlow; the **inference** runs on the Pi using `tflite_runtime` only.

---

## Architecture
**Camera → Pi → Home Assistant (+ Drone)**

```text
Tapo C100 (RTSP) 
    ↓
OpenCV (RTSP capture on Pi)
    ↓
TFLite (MobileNetV3Small)
    ↓                         ┌──────────────┐
thumbs_pi.ai_stream           │  Flask app   │
- classify_frame()            │              │
- overlay_prediction()  ─────▶│  /video      │───▶ Browser / HA camera card
- gesture_callback(label)     │  /api/data   │
                              │  /api/logs   │
                              └──────┬───────┘
                                    │
                         handle_gesture(label)
                          ├─> send webhook to Home Assistant
                          └─> trigger Crazyflie flight pattern (optional)
```

---

## 1. Installation & Setup
This is split into:

1. Runtime on the **Raspberry Pi** (Flask + TFLite + HA webhook + optional drone).
2. Optional **training** environment on PC/Mac for re-training the model.
3. Home Assistant configuration.
4. Optional Crazyflie setup.

---

### 1.1. Pi Runtime (Flask + TFLite + AI Stream)

#### 1.1.1. System requirements

- Raspberry Pi 4 (recommended) with a 64-bit OS (Raspberry Pi OS or similar).
- Python 3.11 installed (`python3.11` available).
- Network access to:
  - Tapo RTSP camera
  - Home Assistant instance
- (Optional) Crazyradio PA dongle + Crazyflie 2.0 drone.

---

#### 1.1.2. Python 3.11 installation (if needed)
If your distro doesn’t ship a new enough Python, you can build 3.11 from source.

##### Update system
    sudo apt update && sudo apt upgrade -y

##### Install build dependencies
    sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev tk-dev uuid-dev wget

##### Go to /usr/src to keep things tidy
    cd /usr/src

##### Download Python 3.11.6 (latest 3.11.x as of now)
    sudo wget https://www.python.org/ftp/python/3.11.6/Python-3.11.6.tgz

##### Extract the archive
    sudo tar -xzf Python-3.11.6.tgz
    cd Python-3.11.6

##### Configure build with optimizations
    sudo ./configure --enable-optimizations

##### Compile using all CPU cores (takes 15–30 min on a Pi 4)
    sudo make -j$(nproc)

##### Install safely (altinstall prevents overwriting system python3)
    sudo make altinstall

##### Check installation
    python3.11 --version

##### Install pip and venv (should already be built-in, but just in case)
    python3.11 -m ensurepip
    python3.11 -m pip install --upgrade pip setuptools wheel

---

#### 1.1.3. Model files
Place your exported model files where thumbs_pi/assets.py expects them, for example:

```text
iot_demo/
  thumbs_pi/
    assets.py
    ai_stream.py
    inference.py
    overlay.py
    models/
      model_int8.tflite
      class_names.txt
```
class_names.txt should contain one label per line, in the same order as training:
```text
thumbs_down
thumbs_up
```
*neutral* is implicit and handled at runtime using a confidence threshold.

---

#### 1.1.4. Environment variables
Configure runtime parameters using environment variables, either via a .env file or directly in your service:

Example .env:

```text
LOG_LEVEL=DEBUG

# Home Assistant
HOMEASSISTANT_TOKEN="your_long_lived_token_here>
HOMEASSISTANT_BASE_URL="http://homeassistant.local:8123"
HA_GESTURE_WEBHOOK_ID="gesture_event"

#Sensors
HA_TEMPERATURE_SENSOR="sensor.co2_monitor_air_quality_detector_air_temperature"
HA_ILLUMINANCE_SENSOR="sensor.multisensor_6_illuminance"
HA_MOTION_SENSOR="binary_sensor.multisensor_6_motion_detection"
HA_CO2_SENSOR="sensor.co2_monitor_air_quality_detector_carbon_dioxide_co2_level"
HA_LIGHT_ENTITY="light.bulb_6_multi_color"

# Camera RTSP
RTSP_URL="rtsp://user:pass@ip/stream"

#MQTT
MQTT_BROKER="homeassistant.local"
MQTT_PORT=1883
MQTT_USERNAME="user"
MQTT_PASSWORD="pass"
```

---

#### 1.1.5. Accessing the Flask app
    python3.11 main.py

Then in a browser: 
```text
http://<pi-ip>:5000/
```
**Current behavior: gesture detection runs as part of the MJPEG generator. As long as some client is consuming /video (browser tab, HA camera card, or even curl http://<pi-ip>:5000/video > /dev/null), the TFLite loop is active and gestures are sent to Home Assistant (+ optionally the drone).**

---

### 1.2. Model Training
To retrain the model, run either:

- model3.py – local/thumb-based training script.
- model3_hagrid.py – variant that uses the HaGRID NPZ dataset.

Typical flow:

1. Run these scripts on a PC/Mac with full TensorFlow (e.g. tensorflow==2.14).

2. Training uses HaGRID NPZ files where:
- Shapes are (N, 3, 224, 224) with dtype=uint8.
- Data is transposed to (N, 224, 224, 3) for Keras.

3. Base model: MobileNetV3Small (ImageNet weights, include_top=False, pooling="avg").

4. Two-stage training:
- Stage 1: train new classification head with frozen base.
- Stage 2: unfreeze part of the base and fine-tune with a smaller learning rate.

5. Export:
- model.keras
- model_int8.tflite (TFLite converter with Optimize.DEFAULT and a representative dataset).
- class_names.txt with labels in the same index order as the model output.

6. Copy model_int8.tflite and class_names.txt to the Pi as described in 1.1.3.

**DISCLAIMER: TENSORFLOW 2.14 IS REQUIRED IF YOU WANT TO RETRAIN THE MODEL. TFLITE IS BASED ON 2.14, AND ANY NEWER OR OLDER VERSION OF TENSORFLOW WILL NOT WORK!**

Install it easily with Pip.
In your venv:
`pip install "tensorflow==2.14"`

---

### 1.3. Home Assistant Setup
The Pi sends webhooks to: `POST http://<home-assistant>/api/webhook/gesture_event`
(Where gesture_event is the value of HA_GESTURE_WEBHOOK_ID.)
The JSON payload looks like:

    { "gesture": "thumbs_up" }

or thumbs_down / neutral.

---

#### 1.3.1. Webhook-triggered automations

Example automations:

**Thumbs up → light on**

    alias: Gesture event - thumbs up
    triggers:
    - trigger: webhook
        webhook_id: gesture_event
        allowed_methods: [POST, PUT]
        local_only: true
    conditions:
    - condition: template
        value_template: "{{ trigger.json.gesture == 'thumbs_up' }}"
    actions:
    - action: light.turn_on
        target:
        entity_id: light.bulb_6_multi_color
        data:
        rgb_color: [38, 162, 105]
        brightness_pct: 60
    mode: single

**Thumbs down → light off**

    alias: Gesture event - thumbs down
    triggers:
    - trigger: webhook
        webhook_id: gesture_event
        allowed_methods: [POST, PUT]
        local_only: true
    conditions:
    - condition: template
        value_template: "{{ trigger.json.gesture == 'thumbs_down' }}"
    actions:
    - action: light.turn_off
        target:
        entity_id: light.bulb_6_multi_color
    mode: single

**Neutral → idle/off behavior**

    alias: Neutral light turn off
    triggers:
    - trigger: webhook
        webhook_id: gesture_event
        allowed_methods: [POST, PUT]
        local_only: true
    conditions:
    - condition: template
        value_template: "{{ trigger.json.gesture == 'neutral' }}"
    actions:
    - action: light.turn_off
        target:
        entity_id: light.bulb_6_multi_color
    mode: single

You can stack this with other automations (motion sensors, CO₂ levels, timeouts, etc.).

---

### 1.4. Optional: Crazyflie 2.0 Drone Setup
If you enable the drone integration, gestures can also trigger a Crazyflie 2.0.

---

#### 1.4.1. Crazyradio & permissions
Add a udev rule (Debian/RPi OS example):

```text
cat << 'EOF' | sudo tee /etc/udev/rules.d/99-crazyradio.rules
SUBSYSTEM=="usb", ATTR{idVendor}=="1915", ATTR{idProduct}=="7777", GROUP="plugdev", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

Unplug/replug the Crazyradio.

---

#### 1.4.2. Find the Crazyflie URI
Use a small scan script (or cfclient) to find the URI, e.g.:
    
    from cflib.crtp import init_drivers, scan_interfaces
    
    init_drivers()
    print("Scanning...")
    for link_uri, _ in scan_interfaces():
        print("Found Crazyflie:", link_uri)

You should get something like:

    Found Crazyflie: radio://0/10/2M

Set this in thumbs_pi/drone_controller.py:

    CF_URI = "radio://0/10/2M"

---

#### 1.4.3. Gesture → drone behavior
thumbs_pi/drone_controller.py exposes a function:

    def handle_gesture(label: str):
        """
        - 'thumbs_up'   -> small hop (raw thrust pattern, tuned HOVER_THRUST)
        - 'thumbs_down' -> controlled descent / land
        - 'neutral'     -> no action
        """

In the Flask app (app.py), a global gesture handler fans out:

    def handle_gesture(label: str):
        logging.info("Global gesture handler: %s", label)
        send_gesture_to_homeassistant(label)  # webhook to HA
        drone_handle_gesture(label)           # Crazyflie reaction

and /video is wired to:

    @app.route("/video")
    def video_feed():
        return mjpeg_response(handle_gesture)

---

# Disclaimer
This project controls real hardware:
- Lights plugged into mains.
- A small but very real flying robot.

Use common sense:
- Test drone patterns at low altitude in a clear space.
- Don’t run “autonomous hop on thumbs_up” unless you’re confident in both model and safety behavior.
- Keep fingers, faces, and cats at a respectful distance from the props.

Have fun responsibly. 🙂
