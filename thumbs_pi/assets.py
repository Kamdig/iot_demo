from __future__ import annotations
import tflite_runtime.interpreter as tflite
from pathlib import Path

# Paths used on the Pi
BASE_DIR = Path(__file__).resolve().parent
TFLITE_QUANT_PATH = BASE_DIR / "model_int8.tflite"
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"

class TFLiteModelBundle:
    def __init__(self, interpreter, input_details, output_details, class_names):
        self.interpreter = interpreter
        self.class_names = class_names

        # Extract input/output metadata
        self.input_index = input_details[0]["index"]
        self.output_index = output_details[0]["index"]

        # Input shape
        _, self.input_height, self.input_width, _ = input_details[0]["shape"]

        # Data types
        self.input_dtype = input_details[0]["dtype"]
        self.output_dtype = output_details[0]["dtype"]

        # Quantization params (if quantized)
        self.input_quant = input_details[0].get("quantization", (0.0, 0))
        self.output_quant = output_details[0].get("quantization", (0.0, 0))


def load_assets(num_threads=2) -> tuple[TFLiteModelBundle, list[str]]:
    """Load the TFLite model + class names. Safe for Raspberry Pi."""

    # Load class names
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Missing class names at {CLASS_NAMES_PATH}")
    class_names = CLASS_NAMES_PATH.read_text().splitlines()

    # Load interpreter
    interpreter = tflite.Interpreter(
        model_path=str(TFLITE_QUANT_PATH),
        num_threads=num_threads
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    bundle = TFLiteModelBundle(
        interpreter=interpreter,
        input_details=input_details,
        output_details=output_details,
        class_names=class_names
    )

    return bundle, class_names


__all__ = ["load_assets", "TFLITE_QUANT_PATH", "CLASS_NAMES_PATH", "TFLiteModelBundle"]
