import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.court_model import CourtGeometryModel


def _simple_model() -> CourtGeometryModel:
    return CourtGeometryModel(
        {
            "kitchen_near_left": [100.0, 300.0],
            "kitchen_near_right": [300.0, 300.0],
            "kitchen_far_left": [100.0, 100.0],
            "kitchen_far_right": [300.0, 100.0],
        }
    )


def test_kitchen_endpoints_preserve_anchor_order():
    model = _simple_model()

    assert model.kitchen_endpoints()["near"] == ((100.0, 300.0), (300.0, 300.0))
    assert model.kitchen_endpoints()["far"] == ((100.0, 100.0), (300.0, 100.0))


def test_boundary_signed_distance_is_positive_on_legal_sides():
    model = _simple_model()

    assert model.left_boundary_line.signed_distance((80.0, 200.0)) > 0
    assert model.left_boundary_line.signed_distance((120.0, 200.0)) < 0

    assert model.right_boundary_line.signed_distance((320.0, 200.0)) > 0
    assert model.right_boundary_line.signed_distance((280.0, 200.0)) < 0


def test_warp_preserves_boundary_sign_convention():
    model = _simple_model()
    transform = np.array(
        [
            [1.0, 0.0, 25.0],
            [0.0, 1.0, -10.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    warped = model.warp(transform)

    assert warped.left_boundary_line.signed_distance((145.0, 190.0)) < 0
    assert warped.left_boundary_line.signed_distance((105.0, 190.0)) > 0
    assert warped.right_boundary_line.signed_distance((305.0, 190.0)) < 0
    assert warped.right_boundary_line.signed_distance((345.0, 190.0)) > 0
