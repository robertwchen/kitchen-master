"""Tests for evaluation utility scaffolding."""

from src.evaluation.metrics import build_confusion_matrix, f1_per_class, precision_per_class, recall_per_class


def test_evaluation_functions_run_without_error() -> None:
    labels = ["behind_line", "on_line", "over_line", "uncertain"]
    y_true = ["behind_line", "on_line", "over_line", "uncertain"]
    y_pred = ["behind_line", "on_line", "uncertain", "uncertain"]

    matrix = build_confusion_matrix(y_true, y_pred, labels)
    precision = precision_per_class(matrix, labels)
    recall = recall_per_class(matrix, labels)
    f1_scores = f1_per_class(precision, recall, labels)

    assert set(matrix.keys()) == set(labels)
    assert set(precision.keys()) == set(labels)
    assert set(recall.keys()) == set(labels)
    assert set(f1_scores.keys()) == set(labels)
