import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile

from src.evaluate import (
    compute_failure_analysis,
    compute_metrics,
    save_confusion_matrix_csv,
    save_failure_analysis_csv,
)
from src.sim_generator import SampleMeta


def _make_meta(scenario, signed_distance, occluded=False, blur=0):
    line_y = 100
    fh = 20
    foot_y = line_y - fh - signed_distance
    return SampleMeta(
        sample_id=0,
        scenario_type=scenario,
        ground_truth_label="legal" if signed_distance > 5 else ("fault" if signed_distance < -2 else "uncertain"),
        foot_x=10,
        foot_y=int(foot_y),
        foot_width=40,
        foot_height=fh,
        line_y=line_y,
        signed_distance_px=signed_distance,
        occlusion_flag=occluded,
        blur_level=blur,
        seed=0,
    )


def test_perfect_predictions():
    labels = ["legal", "fault", "uncertain"]
    m = compute_metrics(labels, labels)
    for label in labels:
        assert m[f"precision_{label}"] == 1.0
        assert m[f"recall_{label}"] == 1.0


def test_uncertain_rate():
    y_true = ["legal", "fault", "legal"]
    y_pred = ["uncertain", "uncertain", "legal"]
    m = compute_metrics(y_true, y_pred)
    assert abs(m["uncertain_rate"] - 2 / 3) < 0.01


def test_false_fault_rate():
    y_true = ["legal", "legal", "fault"]
    y_pred = ["fault", "legal", "fault"]
    m = compute_metrics(y_true, y_pred)
    assert abs(m["false_fault_rate"] - 1 / 3) < 0.01


def test_missed_fault_rate():
    y_true = ["fault", "fault", "legal"]
    y_pred = ["legal", "fault", "legal"]
    m = compute_metrics(y_true, y_pred)
    assert abs(m["missed_fault_rate"] - 1 / 3) < 0.01


def test_confusion_matrix_shape():
    y_true = ["legal", "fault", "uncertain"]
    y_pred = ["legal", "legal", "uncertain"]
    m = compute_metrics(y_true, y_pred)
    assert len(m["confusion_matrix"]) == 3
    assert len(m["confusion_matrix"][0]) == 3


def test_save_confusion_matrix_csv():
    m = compute_metrics(["legal", "fault", "uncertain"], ["legal", "fault", "uncertain"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cm.csv"
        save_confusion_matrix_csv(m, path)
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 4  # header + 3 label rows
        assert "legal" in lines[0]


def test_compute_failure_analysis_grouping():
    metas = [
        _make_meta("clear_legal", 15),
        _make_meta("clear_legal", 15),
        _make_meta("clear_fault", -12),
    ]
    y_pred = ["legal", "uncertain", "fault"]
    rows = compute_failure_analysis(metas, y_pred)
    assert len(rows) >= 2  # at least two scenario groups
    total = sum(r["n_samples"] for r in rows)
    assert total == 3


def test_failure_analysis_accuracy():
    metas = [_make_meta("clear_legal", 15)] * 4
    y_pred = ["legal", "legal", "uncertain", "uncertain"]
    rows = compute_failure_analysis(metas, y_pred)
    assert len(rows) == 1
    row = rows[0]
    assert row["n_samples"] == 4
    assert row["n_correct"] == 2
    assert abs(row["accuracy"] - 0.5) < 0.01


def test_save_failure_analysis_csv():
    metas = [_make_meta("clear_legal", 15), _make_meta("clear_fault", -12)]
    y_pred = ["legal", "fault"]
    rows = compute_failure_analysis(metas, y_pred)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fa.csv"
        save_failure_analysis_csv(rows, path)
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) >= 2  # header + at least one row
