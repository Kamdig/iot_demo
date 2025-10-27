from flask import Flask, jsonify, render_template, request
from app.database.database import database_get_recent
from app.homeassistant.client import set_light_state
import logging
import os

LIGHT_ENTITY = os.getenv("HA_LIGHT_ENTITY", "light.aeotec_led")
LOG_DIR = "logs"

def create_app():
    app = Flask(__name__)

    @app.route('/api/data')
    def get_sensor_data():
        limit = min(request.args.get('limit', default=20, type=int), 100)
        data = database_get_recent("environment", limit)
        return jsonify(data)


    @app.route("/api/light/<state>")
    def toggle_light(state):
        desired_state = state.lower() == "on"
        if not set_light_state(LIGHT_ENTITY, desired_state):
            return jsonify({"status": "error", "light_state": state, "entity": LIGHT_ENTITY}), 500
        return jsonify({"status": "success", "light_state": state, "entity": LIGHT_ENTITY})
    
    @app.route('/api/logs')
    def get_logs():
        log_files = [
            f for f in os.listdir(LOG_DIR)
            if f.endswith(".log")
        ]

        # Sort files by last modified (newest file last)
        log_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f))
        )

        logs = []
        for f in log_files:
            path = os.path.join(LOG_DIR, f)
            try:
                with open(path, "r") as file:
                    # Keep the last 100 lines (oldest → newest)
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
        logging.info("Creating Flask app...")
        return render_template('dashboard.html')

    logging.info("Flask app created.")
    return app
