"""Generate synthetic side-view frames for NVZ foot-fault detection."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _blank_frame(width: int, height: int) -> np.ndarray:
    return np.ones((height, width, 3), dtype=np.uint8) * 220


def _draw_court_line(frame: np.ndarray, line_y: int) -> None:
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (50, 50, 200), 2)


def _draw_foot(frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 120, 30), -1)


def generate_sample(
    cfg: dict,
    rng: np.random.Generator,
    scenario: str,
) -> tuple[np.ndarray, str]:
    """Return (frame, label) for one synthetic sample."""
    W, H = cfg["width"], cfg["height"]
    line_y = int(H * cfg["line_y_frac"])
    fw, fh = cfg["foot_width"], cfg["foot_height"]
    margin = cfg["uncertain_margin_px"]

    frame = _blank_frame(W, H)
    _draw_court_line(frame, line_y)

    foot_x = int(rng.integers(10, W - fw - 10))

    if scenario == "clear_legal":
        foot_y = line_y - fh - int(rng.integers(margin + 2, 30))
        label = "legal"

    elif scenario == "clear_fault":
        foot_y = line_y - fh + int(rng.integers(margin + 2, 20))
        label = "fault"

    elif scenario == "borderline":
        foot_y = line_y - fh + int(rng.integers(-margin, margin + 1))
        label = "uncertain"

    elif scenario == "occluded":
        foot_y = line_y - fh + int(rng.integers(-5, 15))
        label = "uncertain"
        _draw_foot(frame, foot_x, foot_y, fw, fh)
        if cfg.get("blur_scenarios", True):
            kernel = np.zeros((15, 15), dtype=np.float32)
            kernel[7, :] = 1.0 / 15
            frame = cv2.filter2D(frame, -1, kernel)
        return frame, label

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    _draw_foot(frame, foot_x, foot_y, fw, fh)
    return frame, label


def generate_dataset(cfg: dict, seed: int = 42) -> list[tuple[np.ndarray, str]]:
    """Generate a balanced synthetic dataset across all four scenarios."""
    rng = np.random.default_rng(seed)
    scenarios = ["clear_legal", "clear_fault", "borderline", "occluded"]
    per_scenario = cfg["num_samples"] // len(scenarios)

    samples: list[tuple[np.ndarray, str]] = []
    for scenario in scenarios:
        for _ in range(per_scenario):
            samples.append(generate_sample(cfg, rng, scenario))

    indices = rng.permutation(len(samples))
    samples = [samples[i] for i in indices]

    logger.info(f"Generated {len(samples)} synthetic samples")
    return samples
