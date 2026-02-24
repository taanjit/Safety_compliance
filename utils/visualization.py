"""
Visualization utilities for PPE Detection.

Provides helper functions to annotate detection results on images
with color-coded bounding boxes (green = compliant, red = violation).
"""

import cv2
import numpy as np

from config.settings import (
    CLASS_NAMES,
    WORN_PPE_CLASSES,
    MISSING_PPE_CLASSES,
    COLOR_COMPLIANT,
    COLOR_VIOLATION,
    COLOR_NEUTRAL,
)


def get_class_color(class_id: int) -> tuple:
    """Return a BGR color tuple based on the class category.

    Args:
        class_id: Integer class ID from the model output.

    Returns:
        BGR color tuple.
    """
    if class_id in WORN_PPE_CLASSES:
        return COLOR_COMPLIANT
    elif class_id in MISSING_PPE_CLASSES:
        return COLOR_VIOLATION
    else:
        return COLOR_NEUTRAL


def annotate_frame(frame: np.ndarray, results, conf_threshold: float = 0.25) -> np.ndarray:
    """Draw bounding boxes and labels on a frame using detection results.

    Args:
        frame: Input image (BGR, numpy array).
        results: Ultralytics YOLO Results object (single image).
        conf_threshold: Minimum confidence to draw a detection.

    Returns:
        Annotated image (BGR, numpy array).
    """
    annotated = frame.copy()

    if results.boxes is None or len(results.boxes) == 0:
        return annotated

    boxes = results.boxes
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        color = get_class_color(cls_id)
        label = f"{CLASS_NAMES.get(cls_id, f'cls_{cls_id}')} {conf:.2f}"

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


def generate_compliance_summary(results) -> dict:
    """Analyze detection results and produce a compliance summary.

    Args:
        results: Ultralytics YOLO Results object (single image).

    Returns:
        Dictionary with keys:
            - total_persons: Number of persons detected.
            - worn_ppe: Dict of worn PPE class names → counts.
            - missing_ppe: Dict of missing PPE class names → counts.
            - is_compliant: True if no missing PPE was detected.
    """
    summary = {
        "total_persons": 0,
        "worn_ppe": {},
        "missing_ppe": {},
        "is_compliant": True,
    }

    if results.boxes is None or len(results.boxes) == 0:
        return summary

    for box in results.boxes:
        cls_id = int(box.cls[0])

        if cls_id == 6:  # Person
            summary["total_persons"] += 1
        elif cls_id in WORN_PPE_CLASSES:
            name = CLASS_NAMES[cls_id]
            summary["worn_ppe"][name] = summary["worn_ppe"].get(name, 0) + 1
        elif cls_id in MISSING_PPE_CLASSES:
            name = CLASS_NAMES[cls_id]
            summary["missing_ppe"][name] = summary["missing_ppe"].get(name, 0) + 1
            summary["is_compliant"] = False

    return summary
