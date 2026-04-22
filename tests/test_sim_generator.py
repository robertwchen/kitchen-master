import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.sim_generator import generate_dataset, generate_sample

SIM_CFG = {
    "width": 160,
    "height": 120,
    "line_y_frac": 0.6,
    "foot_width": 20,
    "foot_height": 10,
    "uncertain_margin_px": 3,
    "blur_scenarios": False,
}


def test_generate_sample_shapes():
    rng = np.random.default_rng(0)
    for scenario in ["clear_legal", "clear_fault", "borderline", "occluded"]:
        frame, label = generate_sample(SIM_CFG, rng, scenario)
        assert frame.shape == (120, 160, 3), f"Bad shape for {scenario}"
        assert label in ("legal", "fault", "uncertain"), f"Bad label for {scenario}"


def test_clear_legal_label():
    rng = np.random.default_rng(42)
    for _ in range(10):
        _, label = generate_sample(SIM_CFG, rng, "clear_legal")
        assert label == "legal"


def test_clear_fault_label():
    rng = np.random.default_rng(42)
    for _ in range(10):
        _, label = generate_sample(SIM_CFG, rng, "clear_fault")
        assert label == "fault"


def test_generate_dataset_size():
    cfg = {**SIM_CFG, "num_samples": 40}
    samples = generate_dataset(cfg, seed=0)
    assert len(samples) == 40


def test_generate_dataset_label_set():
    cfg = {**SIM_CFG, "num_samples": 40}
    samples = generate_dataset(cfg, seed=0)
    labels = {s[1] for s in samples}
    assert labels <= {"legal", "fault", "uncertain"}
