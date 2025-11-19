from __future__ import annotations
import cv2
import numpy as np
from .assets import TFLiteModelBundle
import math


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Pure NumPy softmax."""
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def _preprocess_frame(frame: np.ndarray, bundle: TFLiteModelBundle) -> np.ndarray:
    """Resize + preprocess a BGR frame for TFLite."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (bundle.input_width, bundle.input_height))

    if bundle.input_dtype == np.uint8:
        # Quantized model: use raw uint8 input
        return resized.astype(np.uint8)

    # Float input → MobileNetV3 scaling: [0,255] → [-1,1]
    arr = resized.astype(np.float32)
    arr = (arr / 127.5) - 1.0
    return arr.astype(np.float32)


def classify_frame(frame: np.ndarray, bundle: TFLiteModelBundle):
    """Run TFLite inference on one frame."""
    img = _preprocess_frame(frame, bundle)
    img = np.expand_dims(img, axis=0)

    # Input quantization (if needed)
    if bundle.input_dtype == np.uint8 and bundle.input_quant[0] != 0:
        scale, zero = bundle.input_quant
        img = (img / scale + zero).astype(np.uint8)

    interpreter = bundle.interpreter
    interpreter.set_tensor(bundle.input_index, img)
    interpreter.invoke()

    output = interpreter.get_tensor(bundle.output_index)[0]

    # Dequantize output if needed
    if bundle.output_dtype == np.uint8 and bundle.output_quant[0] != 0:
        scale, zero = bundle.output_quant
        output = (output.astype(np.float32) - zero) * scale
    else:
        output = output.astype(np.float32)

    probs = _softmax(output)
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    return idx, confidence, probs
