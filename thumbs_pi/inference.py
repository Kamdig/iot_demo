from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .assets import TFLiteModelBundle


def _normalize_frame(frame: np.ndarray, *, width: int, height: int, preprocess: str = "zero_one") -> np.ndarray:
    """Convert BGR frame to RGB, resize and apply preprocessing.

    preprocess options:
      - 'zero_one' : scale to [0, 1] (default)
      - 'mobilenet' : scale to [-1, 1] as MobileNetV3 expects
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (width, height), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32)
    if preprocess == "mobilenet":
        # MobileNetV3 preprocessing: [0,255] -> [-1, 1]
        return (arr / 127.5) - 1.0
    return arr / 255.0


def _quantize_input(data: np.ndarray, *, dtype: np.dtype, scale: float, zero_point: int) -> np.ndarray:
    if scale == 0:
        scale = 1.0
    qmin, qmax = np.iinfo(dtype).min, np.iinfo(dtype).max
    quantized = np.round(data / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(dtype)


def _dequantize_output(data: np.ndarray, *, scale: float, zero_point: int) -> np.ndarray:
    if data.dtype == np.float32 or scale == 0:
        return data.astype(np.float32)
    return (data.astype(np.float32) - zero_point) * scale


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    denom = np.sum(exp_vals)
    if denom == 0:
        return np.zeros_like(exp_vals, dtype=np.float32)
    return (exp_vals / denom).astype(np.float32)


def classify_frame(
    frame: np.ndarray,
    bundle: TFLiteModelBundle,
) -> Tuple[int, float, np.ndarray]:
    """Run a forward pass on an OpenCV BGR frame and return class index plus confidences."""

    normalized = _normalize_frame(
        frame, width=bundle.input_width, height=bundle.input_height, preprocess=getattr(bundle, "preprocess", "zero_one")
    )
    input_tensor = np.expand_dims(normalized, axis=0)

    if bundle.input_dtype == np.float32:
        prepared = input_tensor.astype(np.float32)
    else:
        prepared = _quantize_input(
            input_tensor,
            dtype=bundle.input_dtype,
            scale=bundle.input_quantization[0],
            zero_point=bundle.input_quantization[1],
        )

    interpreter = bundle.interpreter
    interpreter.set_tensor(bundle.input_index, prepared)
    interpreter.invoke()

    output = interpreter.get_tensor(bundle.output_index)[0]
    probabilities = _dequantize_output(
        output,
        scale=bundle.output_quantization[0],
        zero_point=bundle.output_quantization[1],
    )

    probabilities = _softmax(probabilities)
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    return predicted_idx, confidence, probabilities


__all__ = ["classify_frame"]
