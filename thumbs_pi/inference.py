from __future__ import annotations
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from .model3 import IMG_SIZE, load_class_names, TFLITE_QUANT_PATH

# XNNPACK disabled via env var set in main.py
_interpreter = tf.lite.Interpreter(model_path=str(TFLITE_QUANT_PATH))
_interpreter.allocate_tensors()

_input_details = _interpreter.get_input_details()
_output_details = _interpreter.get_output_details()

_input_index = _input_details[0]["index"]
_output_index = _output_details[0]["index"]

_class_names = load_class_names()



def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize and preprocess to match MobileNetV3 expectations."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)

    if _input_dtype == np.uint8:
        # Quantized model
        return resized.astype(np.uint8)

    # Float model → MobileNetV3 preprocess: [0,255] → [-1,1]
    arr = resized.astype(np.float32)
    arr = (arr / 127.5) - 1.0
    return arr.astype(np.float32)


def classify_frame(frame: np.ndarray):
    """Run inference on a BGR frame from OpenCV."""
    img = _preprocess_frame(frame)
    img = np.expand_dims(img, axis=0)

    _interpreter.set_tensor(_input_index, img)
    _interpreter.invoke()

    output = _interpreter.get_tensor(_output_index)[0].astype(np.float32)

    # If output is uint8, convert to float
    if _output_dtype == np.uint8:
        scale = _output_details[0]["quantization_parameters"]["scales"][0]
        zero = _output_details[0]["quantization_parameters"]["zero_points"][0]
        output = (output - zero) * scale

    probs = tf.nn.softmax(output).numpy()
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    return idx, confidence, probs, _class_names[idx]
