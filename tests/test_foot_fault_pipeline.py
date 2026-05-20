import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.court_model import CourtGeometryModel
from src.foot_fault_pipeline import (
    _classify_distance,
    _model_from_reg_row,
    analyze_event_feet,
    infer_active_side,
)


def _model() -> CourtGeometryModel:
    return CourtGeometryModel(
        {
            "kitchen_near_left": [100.0, 300.0],
            "kitchen_near_right": [300.0, 300.0],
            "kitchen_far_left": [100.0, 100.0],
            "kitchen_far_right": [300.0, 100.0],
        }
    )


def test_classify_distance_thresholds():
    assert _classify_distance(20.0, fault_threshold_px=5.0, uncertain_margin_px=15.0) == "legal_volley"
    assert _classify_distance(-6.0, fault_threshold_px=5.0, uncertain_margin_px=15.0) == "foot_fault_volley"
    assert _classify_distance(0.0, fault_threshold_px=5.0, uncertain_margin_px=15.0) == "uncertain"


def test_model_from_registration_row_preserves_right_side_sign():
    model = _model_from_reg_row(
        {
            "kitchen_near_p1_x": "100",
            "kitchen_near_p1_y": "300",
            "kitchen_near_p2_x": "300",
            "kitchen_near_p2_y": "300",
            "kitchen_far_p1_x": "100",
            "kitchen_far_p1_y": "100",
            "kitchen_far_p2_x": "300",
            "kitchen_far_p2_y": "100",
        }
    )

    assert model is not None
    assert model.right_boundary_line.signed_distance((320.0, 200.0)) > 0
    assert model.right_boundary_line.signed_distance((280.0, 200.0)) < 0


def test_infer_active_side_uses_weighted_ball_window():
    side = infer_active_side(
        {
            "frame_index": 10,
            "ball_window": [
                {"frame_index": 9, "ball_x": 260.0, "confidence": 0.8},
                {"frame_index": 10, "ball_x": 330.0, "confidence": 0.9},
                {"frame_index": 11, "ball_x": 340.0, "confidence": 0.9},
            ],
        },
        _model(),
    )

    assert side["active_side"] == "right"
    assert side["active_side_source"] == "ball_window_vote"
    assert side["ball_support_n"] == 3


def test_manual_right_side_event_labels_inside_kitchen_as_fault():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    analysis = analyze_event_feet(
        event={
            "frame_index": 1,
            "active_side_override": "right",
            "override_foot_x": 280.0,
            "override_foot_y": 200.0,
        },
        frame=frame,
        frames=[frame],
        frame_indices=[1],
        frame_index=1,
        model=_model(),
        cfg={
            "fault_threshold_px": 5.0,
            "uncertain_margin_px": 15.0,
            "foot_localizer": {"mode": "background_subtraction"},
        },
    )

    assert analysis["active_side"] == "right"
    assert analysis["label"] == "foot_fault_volley"
    assert analysis["signed_dist_px"] < -5.0


def test_manual_right_side_event_labels_outside_kitchen_as_legal():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    analysis = analyze_event_feet(
        event={
            "frame_index": 1,
            "active_side_override": "right",
            "override_foot_x": 320.0,
            "override_foot_y": 200.0,
        },
        frame=frame,
        frames=[frame],
        frame_indices=[1],
        frame_index=1,
        model=_model(),
        cfg={
            "fault_threshold_px": 5.0,
            "uncertain_margin_px": 15.0,
            "foot_localizer": {"mode": "background_subtraction"},
        },
    )

    assert analysis["active_side"] == "right"
    assert analysis["label"] == "legal_volley"
    assert analysis["signed_dist_px"] > 15.0
