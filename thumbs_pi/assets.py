from __future__ import annotations

from typing import Iterable, Sequence, Tuple
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
import torch.nn as nn
import pickle
import torch

# Project-local paths used by both training and inference code.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
CLASSES_PATH = BASE_DIR / "class_names.pkl"

# Shared transform for evaluation/inference.
INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ConvNeXtClassifier(nn.Module):
    """ConvNeXt Tiny backbone with a frozen body and custom classification head."""

    def __init__(self, num_classes: int) -> None:
        # Initialize the pretrained ConvNeXt Tiny backbone and custom head.
        super().__init__()
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # Freeze backbone weights so only the classifier trains.
        for param in self.convnext.parameters():
            param.requires_grad = False
        self.convnext.classifier[2] = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Defer forward propagation to the wrapped ConvNeXt model.
        return self.convnext(x)


# Decide whether to run on GPU or CPU depending on availability.
def get_device() -> torch.device:
    """Return CUDA device when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Validate that the provided iterable actually contains class labels.
def _ensure_class_names(value: Iterable[str]) -> Sequence[str]:
    class_names = list(value)
    # Guard against accidentally saving an empty class list.
    if not class_names:
        raise ValueError("Class name list is empty; train the model to generate labels.")
    return class_names


# Persist the class label list to disk for later inference.
def save_class_names(class_names: Iterable[str]) -> None:
    """Persist class labels used during training."""
    class_list = _ensure_class_names(class_names)
    CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASSES_PATH.write_bytes(pickle.dumps(class_list))


# Retrieve the serialized class label list for inference.
def load_class_names() -> Sequence[str]:
    """Load class names saved during training."""
    # Require the serialized class file to exist before reading it.
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file missing at {CLASSES_PATH}. Train the model to generate it."
        )
    loaded = pickle.loads(CLASSES_PATH.read_bytes())
    # Ensure the payload matches the expected container format.
    if not isinstance(loaded, (list, tuple)):
        raise TypeError(f"Expected class names to be stored as a list/tuple, got {type(loaded)}.")
    return list(loaded)


# Load the trained weights, label list, and compute device.
def load_assets() -> Tuple[ConvNeXtClassifier, Sequence[str], torch.device]:
    """Load trained model weights and class labels for inference."""
    # Fail immediately if the weights checkpoint is missing.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. Train the model to generate 'best_model.pth'."
        )

    class_names = load_class_names()
    device = get_device()
    model = ConvNeXtClassifier(num_classes=len(class_names)).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, class_names, device


__all__ = [
    "BASE_DIR",
    "MODEL_PATH",
    "CLASSES_PATH",
    "INFERENCE_TRANSFORM",
    "ConvNeXtClassifier",
    "get_device",
    "save_class_names",
    "load_class_names",
    "load_assets",
]