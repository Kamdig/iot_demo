from flask import Flask, jsonify, render_template, request
from thumbs_pi.ai_stream import mjpeg_response
from app.database.database import database_get_recent
from app.homeassistant.client import set_light_state
import requests
import logging
import os

LIGHT_ENTITY = os.getenv("HA_LIGHT_ENTITY", "light.aeotec_led")
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
        url = "http://iotassistant.local:8123/api/webhook/gesture_event"
        data = {"gesture": gesture_name}
        try:
            requests.post(url, json=data, timeout=1)
        except Exception as e:
            print("Failed to send gesture event:", e)

    @app.route("/video")
    def video_feed():
        return mjpeg_response(send_gesture_to_homeassistant)
    
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
