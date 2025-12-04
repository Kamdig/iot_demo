from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
import cv2

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

    # Draw a translucent rectangle so the histogram stays readable regardless of scene brightness.
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
    class_names: Sequence[str],
    label: str,
    confidence: float,
    probabilities: np.ndarray,
    *,
    min_confidence: float,
) -> None:
    """Render the active prediction and probability histogram on the frame."""
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


__all__ = ["draw_probability_panel", "overlay_prediction"]
