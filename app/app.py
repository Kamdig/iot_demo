from flask import Flask, jsonify, request, render_template
from app.database.database import database_get_recent
from app.zwave.zwave import zwave_send_command
import logging

def create_app():
    app = Flask(__name__)

    @app.route('/api/data')
    def get_sensor_data():
        limit = min(request.args.get('limit', default=20, type=int), 100)
        raw_data = database_get_recent("environment", limit=limit)
        
        formatted_data = []
        for entry in raw_data:
            timestamp = entry['timestamp']
            formatted_data.extend([
                {"name": "Temperature (°C)", "value" : entry['temperature'], "timestamp": timestamp},
                {"name": "Illumination (lux)", "value" : entry['illumination'], "timestamp": timestamp},
                {"name": "Motion", "value" : entry ['motion'], "timestamp": timestamp},
                {"name": "CO2 (ppm)", "value" : entry['co2'], "timestamp": timestamp}
            ])
        return jsonify(formatted_data)

    @app.route("/api/light/<state>")
    def toggle_light(state):
        zwave_send_command("aeotec_led", "set", value=state.lower() == "on")
        return jsonify({"status": "success", "light_state": state})

    @app.route('/')
    def dashboard():
        logging.info("Creating Flask app...")
        return render_template('dashboard.html')

    logging.info("Flask app created.")
    return app
