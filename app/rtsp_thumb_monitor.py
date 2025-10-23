"""
Real-time thumbs up/down detector using the ConvNeXt model from thumbs_ai.

Opens the provided RTSP stream, runs inference on incoming frames, and overlays
both the predicted gesture and a per-class probability panel on the video feed.
Press `q` to exit.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
CLASSES_PATH = BASE_DIR / "class_names.pkl"


class ConvNeXtClassifier(nn.Module):
    """Classifier wrapper around ConvNeXt Tiny with a custom head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        for param in self.convnext.parameters():
            param.requires_grad = False
        self.convnext.classifier[2] = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnext(x)


def load_assets() -> Tuple[ConvNeXtClassifier, list[str], torch.device]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Class names not found at {CLASSES_PATH}")

    with CLASSES_PATH.open("rb") as f:
        class_names: list[str] = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNeXtClassifier(num_classes=len(class_names)).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, class_names, device


TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def classify_frame(
    frame: np.ndarray,
    model: ConvNeXtClassifier,
    device: torch.device,
) -> Tuple[int, float, np.ndarray]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    return predicted_idx.item(), confidence.item(), probabilities.cpu().numpy()


def draw_probability_panel(
    frame: np.ndarray,
    class_names: Sequence[str],
    probabilities: np.ndarray,
    *,
    origin: Tuple[int, int] = (10, 70),
    width: int = 280,
    line_height: int = 28,
) -> None:
    """Overlay per-class probability bars onto the frame."""

    if len(class_names) == 0:
        return

    x_start, y_start = origin
    padding = 8
    total_height = line_height * len(class_names) + padding

    panel = frame.copy()
    cv2.rectangle(
        panel,
        (x_start - padding, y_start - line_height),
        (x_start + width, y_start + total_height),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.addWeighted(panel, 0.4, frame, 0.6, 0, frame)

    bar_max_width = width - 150
    bar_start_x = x_start + 110
    score_x = bar_start_x + bar_max_width + 10

    top_idx = int(np.argmax(probabilities))
    for idx, (name, prob) in enumerate(zip(class_names, probabilities)):
        line_y = y_start + idx * line_height
        bar_length = int(bar_max_width * float(prob))
        bar_color = (0, 200, 0) if idx == top_idx else (60, 60, 255)
        cv2.rectangle(
            frame,
            (bar_start_x, line_y - 18),
            (bar_start_x + max(bar_length, 1), line_y - 4),
            bar_color,
            thickness=-1,
        )
        label = f"{name:<12}"
        value = f"{prob * 100:5.1f}%"
        cv2.putText(frame, label, (x_start, line_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, value, (score_x, line_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def overlay_prediction(
    frame: np.ndarray,
    *,
    class_names: Sequence[str],
    label: str,
    confidence: float,
    probabilities: np.ndarray,
    min_confidence: float,
) -> None:
    certainty = f"{confidence * 100:.1f}%"
    if confidence < min_confidence:
        label_display = "uncertain"
        text_color = (0, 255, 255)
    else:
        label_display = label
        text_color = (0, 255, 0) if label == "thumbs_up" else (0, 0, 255)

    overlay = f"{label_display} ({certainty})"
    cv2.putText(
        frame,
        overlay,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        text_color,
        2,
        cv2.LINE_AA,
    )
    draw_probability_panel(frame, class_names, probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time thumbs detection from an RTSP stream.")
    parser.add_argument(
        "--rtsp-url",
        default="rtsp://iotworldcam:smart123@10.136.171.24/stream2",
        help="RTSP stream URL to connect to.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Process every Nth frame to reduce load (default: 2).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence required to report a gesture (default: 0.6).",
    )
    args = parser.parse_args()

    try:
        model, class_names, device = load_assets()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(f"[INFO] Loaded model on device: {device}")
    print(f"[INFO] Class labels: {class_names}")

    cap = cv2.VideoCapture(args.rtsp_url)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open RTSP stream: {args.rtsp_url}")
        sys.exit(1)

    print("[INFO] Press 'q' to exit.")
    frame_idx = 0
    last_label = None
    last_report_time = 0.0
    last_prediction = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame from stream. Retrying...")
                time.sleep(0.5)
                continue

            frame_idx += 1
            should_classify = frame_idx % max(args.frame_skip, 1) == 0

            if should_classify:
                predicted_idx, confidence, probabilities = classify_frame(frame, model, device)
                label = class_names[predicted_idx]
                last_prediction = (label, confidence, probabilities)

                now = time.time()
                if confidence >= args.min_confidence and (
                    label != last_label or now - last_report_time > 2.0
                ):
                    prob_str = ", ".join(
                        f"{name}: {prob * 100:.1f}%"
                        for name, prob in zip(class_names, probabilities)
                    )
                    certainty = f"{confidence * 100:.1f}%"
                    print(f"[{time.strftime('%H:%M:%S')}] Detected {label} ({certainty}) | {prob_str}")
                    last_label = label
                    last_report_time = now

            if last_prediction is not None:
                overlay_prediction(
                    frame,
                    class_names=class_names,
                    label=last_prediction[0],
                    confidence=last_prediction[1],
                    probabilities=last_prediction[2],
                    min_confidence=args.min_confidence,
                )

            cv2.imshow("Thumbs Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
