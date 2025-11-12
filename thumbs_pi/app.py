from flask import Flask
from ai_stream import mjpeg_response

app = Flask(__name__)

@app.route("/video")
def video_feed():
    return mjpeg_response()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
