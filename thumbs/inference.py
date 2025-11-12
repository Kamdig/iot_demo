from __future__ import annotations

from .assets import ConvNeXtClassifier, INFERENCE_TRANSFORM
from typing import Tuple
from PIL import Image
import numpy as np
import torch
import cv2


# Convert an OpenCV frame into a model prediction and probability vector.
def classify_frame(
    frame: np.ndarray,
    model: ConvNeXtClassifier,
    device: torch.device,
) -> Tuple[int, float, np.ndarray]:
    """Run a forward pass on an OpenCV BGR frame and return class index plus confidences."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    return predicted_idx.item(), confidence.item(), probabilities.cpu().numpy()


__all__ = ["classify_frame"]
