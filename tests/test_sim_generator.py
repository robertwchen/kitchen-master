import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.sim_generator import SampleMeta, generate_dataset, generate_sample

SIM_CFG = {
    "width": 160,
    "height": 120,
    "line_y_frac": 0.6,
    "foot_width": 20,
    "foot_height": 10,
    "uncertain_margin_px": 3,
    "blur_scenarios": False,
}


def test_generate_sample_returns_meta():
    rng = np.random.default_rng(0)
    frame, meta = generate_sample(SIM_CFG, rng, "clear_legal", sample_id=7, seed=99)
    assert frame.shape == (120, 160, 3)
    assert isinstance(meta, SampleMeta)
    assert meta.sample_id == 7
    assert meta.seed == 99
    assert meta.scenario_type == "clear_legal"
    assert meta.ground_truth_label == "legal"


def test_meta_fields_present():
    rng = np.random.default_rng(1)
    _, meta = generate_sample(SIM_CFG, rng, "clear_fault", sample_id=0, seed=0)
    assert hasattr(meta, "foot_x")
    assert hasattr(meta, "foot_y")
    assert hasattr(meta, "foot_width")
    assert hasattr(meta, "foot_height")
    assert hasattr(meta, "line_y")
    assert hasattr(meta, "signed_distance_px")
    assert hasattr(meta, "occlusion_flag")
    assert hasattr(meta, "blur_level")


def test_signed_distance_legal():
    rng = np.random.default_rng(42)
    for _ in range(10):
        _, meta = generate_sample(SIM_CFG, rng, "clear_legal")
        assert meta.signed_distance_px > 0, "Legal sample must have positive distance"


def test_signed_distance_fault():
    rng = np.random.default_rng(42)
    for _ in range(10):
        _, meta = generate_sample(SIM_CFG, rng, "clear_fault")
        assert meta.signed_distance_px < 0, "Fault sample must have negative distance"


def test_signed_distance_consistency():
    rng = np.random.default_rng(5)
    _, meta = generate_sample(SIM_CFG, rng, "clear_legal")
    expected = meta.line_y - (meta.foot_y + meta.foot_height)
    assert meta.signed_distance_px == expected


def test_occlusion_flag_occluded():
    rng = np.random.default_rng(0)
    _, meta = generate_sample(SIM_CFG, rng, "occluded")
    assert meta.occlusion_flag is True


def test_occlusion_flag_clear():
    rng = np.random.default_rng(0)
    for scenario in ["clear_legal", "clear_fault", "borderline"]:
        _, meta = generate_sample(SIM_CFG, rng, scenario)
        assert meta.occlusion_flag is False


def test_blur_level_with_blur():
    cfg = {**SIM_CFG, "blur_scenarios": True}
    rng = np.random.default_rng(0)
    _, meta = generate_sample(cfg, rng, "occluded")
    assert meta.blur_level == 1


def test_blur_level_no_blur():
    rng = np.random.default_rng(0)
    _, meta = generate_sample(SIM_CFG, rng, "occluded")
    assert meta.blur_level == 0


def test_generate_dataset_size():
    cfg = {**SIM_CFG, "num_samples": 40}
    samples = generate_dataset(cfg, seed=0)
    assert len(samples) == 40


def test_generate_dataset_meta():
    cfg = {**SIM_CFG, "num_samples": 40}
    samples = generate_dataset(cfg, seed=0)
    metas = [m for _, m in samples]
    labels = {m.ground_truth_label for m in metas}
    assert labels <= {"legal", "fault", "uncertain"}
    scenarios = {m.scenario_type for m in metas}
    assert scenarios == {"clear_legal", "clear_fault", "borderline", "occluded"}


def test_generate_dataset_unique_ids():
    cfg = {**SIM_CFG, "num_samples": 40}
    samples = generate_dataset(cfg, seed=0)
    ids = [m.sample_id for _, m in samples]
    assert len(set(ids)) == len(ids), "sample_ids must be unique"
