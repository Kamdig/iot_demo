import cv2
import os
import time
from pathlib import Path

# === SETTINGS ===
RTSP_URL = "rtsp://iotworldcam:smart123@192.168.1.204/stream2"
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "train"

# Class → key mapping
CLASSES = {
    "neutral": "0",
    "thumbs_up": "1",
    "thumbs_down": "2",
    "left": "3",
    "right": "4",
}

# Create folders
for label in CLASSES:
    folder = DATASET_DIR / label
    folder.mkdir(parents=True, exist_ok=True)

print("=== Dataset Capture ===")
print("Press keys to capture images:")
for cls, key in CLASSES.items():
    print(f"  {key}: {cls}")
print("Press q to quit.")
print("=========================")

cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise RuntimeError(f"Could not open RTSP stream: {RTSP_URL}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame... retrying.")
        time.sleep(0.2)
        continue

    cv2.imshow("Dataset Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # Check which class key was pressed
    for cls, cls_key in CLASSES.items():
        if key == ord(cls_key):
            filename = f"{int(time.time() * 1000)}.jpg"
            filepath = DATASET_DIR / cls / filename
            cv2.imwrite(str(filepath), frame)
            print(f"Saved {filepath}")
            break

cap.release()
cv2.destroyAllWindows()
