from flask import Flask, jsonify, render_template, request
from app.database.database import database_get_recent
from app.homeassistant.client import set_light_state
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


    @app.route("/api/light/<state>")
    def toggle_light(state):
        # Flip the configured Home Assistant light on/off.
        desired_state = state.lower() == "on"
        # Report failures if Home Assistant rejected the command.
        if not set_light_state(LIGHT_ENTITY, desired_state):
            return jsonify({"status": "error", "light_state": state, "entity": LIGHT_ENTITY}), 500
        return jsonify({"status": "success", "light_state": state, "entity": LIGHT_ENTITY})
    
    
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
