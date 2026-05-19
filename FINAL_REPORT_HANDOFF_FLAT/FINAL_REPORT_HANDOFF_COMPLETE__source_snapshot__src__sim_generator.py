"""Generate synthetic side-view frames for NVZ foot-fault detection."""

import csv
import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SampleMeta:
    sample_id: int
    scenario_type: str
    ground_truth_label: str
    foot_x: int
    foot_y: int
    foot_width: int
    foot_height: int
    line_y: int
    signed_distance_px: int  # line_y - foot_bottom; +ve = legal, -ve = fault
    occlusion_flag: bool
    blur_level: int           # 0 = none, 1 = motion blur
    seed: int
    frame_path: str = ""


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
    sample_id: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, SampleMeta]:
    """Return (frame, SampleMeta) for one synthetic sample."""
    W, H = cfg["width"], cfg["height"]
    line_y = int(H * cfg["line_y_frac"])
    fw, fh = cfg["foot_width"], cfg["foot_height"]
    margin = cfg["uncertain_margin_px"]

    frame = _blank_frame(W, H)
    _draw_court_line(frame, line_y)

    foot_x = int(rng.integers(10, W - fw - 10))
    occlusion_flag = False
    blur_level = 0

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
        occlusion_flag = True
        _draw_foot(frame, foot_x, foot_y, fw, fh)
        if cfg.get("blur_scenarios", True):
            kernel = np.zeros((15, 15), dtype=np.float32)
            kernel[7, :] = 1.0 / 15
            frame = cv2.filter2D(frame, -1, kernel)
            blur_level = 1
        return frame, SampleMeta(
            sample_id=sample_id,
            scenario_type=scenario,
            ground_truth_label=label,
            foot_x=foot_x,
            foot_y=int(foot_y),
            foot_width=fw,
            foot_height=fh,
            line_y=line_y,
            signed_distance_px=line_y - (int(foot_y) + fh),
            occlusion_flag=occlusion_flag,
            blur_level=blur_level,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    _draw_foot(frame, foot_x, foot_y, fw, fh)
    return frame, SampleMeta(
        sample_id=sample_id,
        scenario_type=scenario,
        ground_truth_label=label,
        foot_x=foot_x,
        foot_y=int(foot_y),
        foot_width=fw,
        foot_height=fh,
        line_y=line_y,
        signed_distance_px=line_y - (int(foot_y) + fh),
        occlusion_flag=occlusion_flag,
        blur_level=blur_level,
        seed=seed,
    )


def generate_dataset(cfg: dict, seed: int = 42) -> list[tuple[np.ndarray, SampleMeta]]:
    """Generate a balanced synthetic dataset across all four scenarios."""
    master_rng = np.random.default_rng(seed)
    scenarios = ["clear_legal", "clear_fault", "borderline", "occluded"]
    per_scenario = cfg["num_samples"] // len(scenarios)

    samples: list[tuple[np.ndarray, SampleMeta]] = []
    sample_id = 0
    for scenario in scenarios:
        for _ in range(per_scenario):
            sample_seed = int(master_rng.integers(0, 2**31))
            sample_rng = np.random.default_rng(sample_seed)
            frame, meta = generate_sample(cfg, sample_rng, scenario, sample_id=sample_id, seed=sample_seed)
            samples.append((frame, meta))
            sample_id += 1

    indices = master_rng.permutation(len(samples))
    samples = [samples[i] for i in indices]

    logger.info(f"Generated {len(samples)} synthetic samples")
    return samples


def save_metadata_csv(metas: list[SampleMeta], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not metas:
        return
    fields = [f.name for f in dataclasses.fields(SampleMeta)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in metas:
            writer.writerow(dataclasses.asdict(m))
    logger.info(f"Saved metadata to {path} ({len(metas)} rows)")
