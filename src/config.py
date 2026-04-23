import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {path}")
    return cfg


def get_default_config() -> dict:
    return {
        "run_name": "sim_v1",
        "sim": {
            "num_samples": 200,
            "width": 320,
            "height": 240,
            "line_y_frac": 0.6,
            "foot_width": 40,
            "foot_height": 20,
            "uncertain_margin_px": 5,
            "blur_scenarios": True,
        },
        "detector": {
            "fault_threshold_px": 2,
            "uncertain_margin_px": 8,
            "line_detection": "hough",
        },
        "output": {
            "results_dir": "results/",
            "save_plots": True,
            "save_frames": False,
            "save_overlays": True,
        },
    }
