from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Optional
import pickle
import numpy as np

# --- Optional runtime dependencies -----------------------------------------------------------
try:  # Prefer the lightweight runtime on Raspberry Pi builds
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except ImportError:  # pragma: no cover - fallback when tflite-runtime is unavailable
    try:
        from tensorflow.lite import Interpreter  # type: ignore
    except ImportError:  # pragma: no cover - surfaced at load-time below
        Interpreter = None  # type: ignore

# Project-local paths used by both training and inference code.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
CLASSES_PATH = BASE_DIR / "class_names.pkl"
CLASSES_TXT_PATH = BASE_DIR / "class_names.txt"

TFLITE_CANDIDATES = (
    BASE_DIR / "model_int8.tflite",
    BASE_DIR / "model_quant.tflite",
    BASE_DIR / "model.tflite",
    # Common MobileNetV3-small names we might provide for Raspberry Pi deployments
    BASE_DIR / "mobilenet_v3_small.tflite",
    BASE_DIR / "mobilenet_v3_small_int8.tflite",
)

# Legacy placeholder kept for backward compatibility with the thumbs package API.
INFERENCE_TRANSFORM = None  # type: ignore


@dataclass
class TFLiteModelBundle:
    """Container for a TensorFlow Lite interpreter plus cached IO metadata."""

    interpreter: "Interpreter"
    input_index: int
    output_index: int
    input_height: int
    input_width: int
    input_dtype: np.dtype
    input_quantization: Tuple[float, int]
    output_quantization: Tuple[float, int]
    # Indicates which preprocessing the model expects for float inputs.
    # For example: 'mobilenet' means inputs should be scaled to [-1, 1].
    preprocess: str = "zero_one"


def _ensure_class_names(value: Iterable[str]) -> Sequence[str]:
    class_names = list(value)
    if not class_names:
        raise ValueError("Class name list is empty; train the model to generate labels.")
    return class_names


def save_class_names(class_names: Iterable[str]) -> None:
    """Persist class labels used during training."""
    class_list = list(_ensure_class_names(class_names))
    CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASSES_PATH.write_bytes(pickle.dumps(class_list))
    CLASSES_TXT_PATH.write_text("\n".join(class_list), encoding="utf-8")


def load_class_names() -> Sequence[str]:
    """Load class names saved during training."""
    if CLASSES_TXT_PATH.exists():
        entries = [line.strip() for line in CLASSES_TXT_PATH.read_text(encoding="utf-8").splitlines()]
        class_names = [entry for entry in entries if entry]
        if class_names:
            return class_names
    if CLASSES_PATH.exists():
        loaded = pickle.loads(CLASSES_PATH.read_bytes())
        if not isinstance(loaded, (list, tuple)):
            raise TypeError(
                f"Expected class names to be stored as a list/tuple, got {type(loaded)}."
            )
        return list(loaded)
    raise FileNotFoundError(
        "Class names file missing. Train the model to generate 'class_names.txt' or '.pkl'."
    )


def _resolve_tflite_path(model_path: Optional[Path] = None) -> Path:
    if model_path is not None:
        resolved = Path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(f"TFLite model not found at {resolved}.")
        return resolved
    for candidate in TFLITE_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in TFLITE_CANDIDATES)
    raise FileNotFoundError(f"No TensorFlow Lite model found. Looked for: {searched}")


def _assert_interpreter_available() -> type:
    if Interpreter is None:  # pragma: no cover - triggered when runtime missing
        raise ImportError(
            "TensorFlow Lite runtime is not installed. Install 'tflite-runtime' or 'tensorflow'."
        )
    return Interpreter


def _to_quant_tuple(raw: Tuple[float, int] | Tuple[float, int, int]) -> Tuple[float, int]:
    if len(raw) >= 2:
        return float(raw[0]), int(raw[1])
    return 0.0, 0


def load_assets(model_path: Optional[Path] = None, *, num_threads: Optional[int] = 4) -> Tuple[TFLiteModelBundle, Sequence[str]]:
    """Load a TensorFlow Lite interpreter and class labels for inference on constrained devices.

    Parameters
    ----------
    model_path:
        Optional Path to a specific .tflite file. If omitted, the function will search
        the TFLITE_CANDIDATES tuple.
    num_threads:
        Number of threads to request from the TFLite interpreter (helps on Raspberry Pi 4).
        If the underlying Interpreter class does not accept the argument it will be ignored.
    """

    interpreter_cls = _assert_interpreter_available()
    resolved_model = _resolve_tflite_path(model_path)

    # Try to instantiate the interpreter with multi-threading enabled where supported.
    try:
        # Many tflite-runtime builds accept num_threads as a kwarg.
        interpreter = interpreter_cls(model_path=str(resolved_model), num_threads=int(num_threads or 1))
    except TypeError:
        # Fallback for interpreter implementations that don't accept num_threads.
        interpreter = interpreter_cls(model_path=str(resolved_model))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError("The TFLite model must have exactly one input and one output tensor.")

    input_detail = input_details[0]
    output_detail = output_details[0]
    _, height, width, *_ = input_detail["shape"]

    bundle = TFLiteModelBundle(
        interpreter=interpreter,
        input_index=int(input_detail["index"]),
        output_index=int(output_detail["index"]),
        input_height=int(height),
        input_width=int(width),
        input_dtype=np.dtype(input_detail["dtype"]),
        input_quantization=_to_quant_tuple(tuple(input_detail.get("quantization", (0.0, 0)))),
        output_quantization=_to_quant_tuple(tuple(output_detail.get("quantization", (0.0, 0)))),
    )

    # Heuristic: if the model filename contains 'mobilenet' assume MobileNet preprocessing
    model_name = resolved_model.name.lower()
    if "mobilenet" in model_name or "mobilenet_v3" in model_name or "mobilenetv3" in model_name:
        bundle.preprocess = "mobilenet"

    class_names = load_class_names()
    return bundle, class_names


__all__ = [
    "BASE_DIR",
    "MODEL_PATH",
    "CLASSES_PATH",
    "CLASSES_TXT_PATH",
    "TFLITE_CANDIDATES",
    "INFERENCE_TRANSFORM",
    "TFLiteModelBundle",
    "save_class_names",
    "load_class_names",
    "load_assets",
]
