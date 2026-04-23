"""Visualization utilities for court registration overlays."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Color palette (BGR)
COLOR_NEAR = (0, 220, 50)      # green  — near kitchen line
COLOR_FAR  = (0, 180, 255)     # orange — far kitchen line
COLOR_LEGAL_FILL = (0, 255, 0) # green fill for legal zone
COLOR_TEXT = (255, 255, 255)


def draw_kitchen_lines(
    frame: np.ndarray,
    registration,              # CourtRegistration instance
    thickness: int = 2,
    label: bool = True,
) -> np.ndarray:
    """Draw registered kitchen lines on a copy of frame."""
    out = frame.copy()
    H, W = out.shape[:2]

    for line_name, line, color in [
        ("near NVZ", registration.near_line, COLOR_NEAR),
        ("far NVZ",  registration.far_line,  COLOR_FAR),
    ]:
        if line is None:
            continue
        pt1, pt2 = line.endpoints_in_frame(W, H)
        cv2.line(out, pt1, pt2, color, thickness)
        if label:
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            cv2.putText(out, line_name, (mid_x - 40, mid_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return out


def mark_legal_zone(
    frame: np.ndarray,
    registration,
    alpha: float = 0.12,
) -> np.ndarray:
    """
    Shade the legal zone (behind the near kitchen line) with a transparent fill.
    Requires registration.legal_ref_point to determine which side is legal.
    """
    if registration.near_line is None or registration.legal_ref_point is None:
        return frame

    sign = registration.legal_side_sign()
    if sign is None:
        return frame

    H, W = frame.shape[:2]
    line = registration.near_line

    # Vectorised signed-distance computation over all pixels
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist = line.a * xx + line.b * yy + line.c
    mask = (sign * dist > 0)

    overlay = frame.copy()
    fill = np.array(COLOR_LEGAL_FILL, dtype=np.float32)
    overlay[mask] = (
        overlay[mask].astype(np.float32) * (1 - alpha) + fill * alpha
    ).astype(np.uint8)

    return overlay


def draw_frame_info(
    frame: np.ndarray,
    frame_index: int,
    timestamp_s: float,
    extra: str = "",
) -> np.ndarray:
    """Stamp frame index and timestamp in the top-left corner."""
    out = frame.copy()
    text = f"f={frame_index}  t={timestamp_s:.2f}s"
    if extra:
        text += f"  {extra}"
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    return out


def export_debug_frame(
    frame: np.ndarray,
    path: Path,
    registration=None,
    frame_index: int = 0,
    timestamp_s: float = 0.0,
    mark_legal: bool = True,
) -> None:
    """Save a single annotated debug frame as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    if registration is not None:
        if mark_legal:
            out = mark_legal_zone(out, registration)
        out = draw_kitchen_lines(out, registration, thickness=2, label=True)
    out = draw_frame_info(out, frame_index, timestamp_s)
    cv2.imwrite(str(path), out)
    logger.info(f"Debug frame saved: {path}")


def write_overlay_video(
    video_path: Path,
    out_path: Path,
    registration,
    fps: float = 10.0,
    scale: float = 0.5,
    frame_step: int = 1,
    mark_legal: bool = True,
) -> None:
    """
    Write an annotated overlay video from the source clip.

    Args:
        scale:      output resolution scale (0.5 = half resolution)
        frame_step: process every Nth frame (use >1 to speed up)
        fps:        output video FPS
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            ts = frame_idx / src_fps
            small = cv2.resize(frame, (W, H))
            if mark_legal:
                small = mark_legal_zone(small, registration)
            small = draw_kitchen_lines(small, registration, thickness=2, label=True)
            small = draw_frame_info(small, frame_idx, ts)
            writer.write(small)
            written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info(f"Overlay video saved: {out_path}  ({written} frames, {fps}fps, {scale}x scale)")
