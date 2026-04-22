import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import compute_metrics


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
