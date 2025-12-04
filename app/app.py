from thumbs_pi.drone_controller import handle_gesture as drone_handle_gesture
from flask import Flask, jsonify, render_template, request
from app.database.database import database_get_recent
from thumbs_pi.ai_stream import mjpeg_response
import requests
import logging
import os

HOME_ASSISTANT_BASE_URL = os.getenv("HOMEASSISTANT_BASE_URL", "").rstrip("/")
GESTURE_WEBHOOK_ID = os.getenv("HA_GESTURE_WEBHOOK_ID", "gesture_event")
LOG_DIR = "logs"

def create_app():
    # Flask application factory: wires up API routes and templates.
    app = Flask(__name__)

    @app.route('/api/data')
    def get_sensor_data():
        # Return the most recent sensor samples, capped to 100 items.
        limit = min(request.args.get('limit', default=20, type=int), 100)
        data = database_get_recent("environment", limit)
        return jsonify(data)

    def send_gesture_to_homeassistant(gesture_name):
        if not HOME_ASSISTANT_BASE_URL:
            logging.warning("HOMEASSISTANT_BASE_URL not set; skipping gesture webhook.")
            return

        url = f"{HOME_ASSISTANT_BASE_URL}/api/webhook/{GESTURE_WEBHOOK_ID}"
        data = {"gesture": gesture_name}

        try:
            resp = requests.post(url, json=data, timeout=2)
            resp.raise_for_status()
            logging.debug(
                "Sent gesture '%s' to %s (status=%s)",
                gesture_name, url, resp.status_code
            )
        except Exception as e:
            logging.exception(
                "Failed to send gesture event '%s' to %s: %s",
                gesture_name, url, e
            )

    def handle_gesture(label: str):
        """
        Fan-out for every detected gesture:
        - send to Home Assistant
        - trigger Crazyflie pattern
        """
        logging.info("Global gesture handler: %s", label)

        # 1) Home Assistant
        send_gesture_to_homeassistant(label)

        # 2) Drone
        drone_handle_gesture(label)

    @app.route("/video")
    def video_feed():
        return mjpeg_response(handle_gesture)
    
    @app.route('/api/logs')
    def get_logs():
        base_dir = os.path.abspath(os.path.dirname(__file__))
        log_dir = os.path.join(base_dir, "..", LOG_DIR)
        log_dir = os.path.abspath(log_dir)

        # Gather the *.log files from the logs directory.
        log_files = [
            f for f in os.listdir(log_dir)
            if f.endswith(".log")
        ]

        # Sort by last modified time (newest file first)
        log_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
            reverse=False  # change to True if you want newest first
        )

        logs = []
        # Iterate over each log file and capture a safe preview of the contents.
        for f in log_files:
            path = os.path.join(log_dir, f)
            try:
                # Skip enormous files (>5 MB) for safety
                if os.path.getsize(path) > 5 * 1024 * 1024:
                    logs.append({
                        "filename": f,
                        "content": f"Skipped: {f} is too large (>5MB)."
                    })
                    continue

                with open(path, "r", encoding="utf-8", errors="replace") as file:
                    # Keep the last 100 lines only
                    lines = file.readlines()[-100:]
                    logs.append({
                        "filename": f,
                        "content": "".join(lines)
                    })
            except Exception as e:
                logs.append({
                    "filename": f,
                    "content": f"Error reading {f}: {e}"
                })
        return jsonify(logs)


    @app.route('/')
    def dashboard():
        # Serve the main dashboard template.
        logging.info("Creating Flask app...")
        return render_template('dashboard.html')

    logging.info("Flask app created.")
    return app
