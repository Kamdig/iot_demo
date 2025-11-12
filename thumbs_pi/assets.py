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

try:  # PyTorch is only required for the legacy training helpers in this module
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
except ImportError:  # pragma: no cover - allows inference-only deployments without torch
    torch = None  # type: ignore
    nn = None  # type: ignore
    models = None  # type: ignore
    transforms = None  # type: ignore

# Project-local paths used by both training and inference code.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
CLASSES_PATH = BASE_DIR / "class_names.pkl"
CLASSES_TXT_PATH = BASE_DIR / "class_names.txt"

TFLITE_CANDIDATES = (
    BASE_DIR / "model_int8.tflite",
    BASE_DIR / "model_quant.tflite",
    BASE_DIR / "model.tflite",
)

# Shared transform for evaluation/inference.
if transforms is not None:
    INFERENCE_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
else:  # pragma: no cover - training utilities are unavailable without torchvision
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


if torch is not None and nn is not None and models is not None:

    class ConvNeXtClassifier(nn.Module):
        """ConvNeXt Tiny backbone with a frozen body and custom classification head."""

        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            for param in self.convnext.parameters():
                param.requires_grad = False
            self.convnext.classifier[2] = nn.Linear(768, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.convnext(x)


    def get_device() -> "torch.device":  # type: ignore[override]
        """Return CUDA device when available, otherwise fall back to CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:  # pragma: no cover - executed only when torch is absent

    class ConvNeXtClassifier:  # type: ignore[no-redef]
        def __init__(self, *_: object, **__: object) -> None:
            raise ImportError("PyTorch is not installed; ConvNeXtClassifier is unavailable.")


    def get_device() -> "torch.device":  # type: ignore[override]
        raise ImportError("PyTorch is not installed; get_device() cannot be used.")


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


def load_assets(model_path: Optional[Path] = None) -> Tuple[TFLiteModelBundle, Sequence[str]]:
    """Load a TensorFlow Lite interpreter and class labels for inference on constrained devices."""

    interpreter_cls = _assert_interpreter_available()
    resolved_model = _resolve_tflite_path(model_path)

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

    class_names = load_class_names()
    return bundle, class_names


__all__ = [
    "BASE_DIR",
    "MODEL_PATH",
    "CLASSES_PATH",
    "CLASSES_TXT_PATH",
    "TFLITE_CANDIDATES",
    "INFERENCE_TRANSFORM",
    "ConvNeXtClassifier",
    "TFLiteModelBundle",
    "get_device",
    "save_class_names",
    "load_class_names",
    "load_assets",
]
