from __future__ import annotations
from .assets import TFLiteModelBundle
import numpy as np
import cv2

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Pure NumPy softmax."""
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def _preprocess_frame(frame: np.ndarray, bundle: TFLiteModelBundle) -> np.ndarray:
    """Resize + preprocess a BGR frame for TFLite."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (bundle.input_width, bundle.input_height))

    if bundle.input_dtype == np.uint8:
        # Quantized model: raw 0–255 uint8, model handles quantization
        return resized.astype(np.uint8)

    # Float input: just 0–255 float32, model contains preprocess_input
    return resized.astype(np.float32)


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

    probs = output
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    return idx, confidence, probs
