"""Baseline NVZ line-contact detector using classical CV."""

import logging
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Label = Literal["legal", "fault", "uncertain"]


def detect_line_y(frame: np.ndarray, method: str = "hough") -> int | None:
    """Estimate y-coordinate of the kitchen line. Returns None if not found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    if method == "hough":
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=50,
            minLineLength=frame.shape[1] // 3,
            maxLineGap=20,
        )
        if lines is None:
            return None
        horizontal = [l[0] for l in lines if abs(l[0][1] - l[0][3]) < 5]
        if not horizontal:
            return None
        ys = [(l[1] + l[3]) // 2 for l in horizontal]
        return int(np.median(ys))

    if method == "gradient":
        col_means = edges.mean(axis=1)
        return int(np.argmax(col_means))

    raise ValueError(f"Unknown method: {method}")


def detect_foot_bottom(frame: np.ndarray) -> int | None:
    """Estimate y-coordinate of the bottom edge of the foot region."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Green range tuned for sim; adapt hue/saturation bounds for real footage
    mask = cv2.inRange(hsv, (35, 40, 20), (85, 255, 200))
    rows_with_foot = np.where(mask.any(axis=1))[0]
    if len(rows_with_foot) == 0:
        return None
    return int(rows_with_foot.max())


def classify(
    line_y: int | None,
    foot_bottom: int | None,
    fault_threshold_px: int,
    uncertain_margin_px: int,
) -> Label:
    """
    Classify based on gap between foot bottom and line.
    Positive gap = foot above line (legal); negative = foot crosses line (fault).
    """
    if line_y is None or foot_bottom is None:
        return "uncertain"

    gap = line_y - foot_bottom

    if gap > uncertain_margin_px:
        return "legal"
    if gap < -fault_threshold_px:
        return "fault"
    return "uncertain"


def predict(frame: np.ndarray, cfg: dict) -> Label:
    """Run full detection pipeline on a single frame."""
    line_y = detect_line_y(frame, method=cfg.get("line_detection", "hough"))
    foot_bottom = detect_foot_bottom(frame)
    return classify(
        line_y,
        foot_bottom,
        fault_threshold_px=cfg.get("fault_threshold_px", 2),
        uncertain_margin_px=cfg.get("uncertain_margin_px", 8),
    )


def predict_with_details(frame: np.ndarray, cfg: dict) -> dict:
    """Return label plus intermediate detector outputs for diagnostics."""
    line_y = detect_line_y(frame, method=cfg.get("line_detection", "hough"))
    foot_bottom = detect_foot_bottom(frame)
    label = classify(
        line_y,
        foot_bottom,
        fault_threshold_px=cfg.get("fault_threshold_px", 2),
        uncertain_margin_px=cfg.get("uncertain_margin_px", 8),
    )
    gap = (line_y - foot_bottom) if (line_y is not None and foot_bottom is not None) else None
    return {
        "label": label,
        "detected_line_y": line_y,
        "detected_foot_bottom": foot_bottom,
        "detected_gap_px": gap,
    }
