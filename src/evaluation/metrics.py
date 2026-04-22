"""Minimal evaluation utilities for v1 classification research."""

from __future__ import annotations

from typing import Dict, Iterable, List


def build_confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: List[str],
) -> Dict[str, Dict[str, int]]:
    """Build a nested-dict confusion matrix keyed by true then predicted label."""
    matrix = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label not in matrix:
            continue
        if pred_label not in matrix[true_label]:
            continue
        matrix[true_label][pred_label] += 1

    return matrix


def precision_per_class(confusion_matrix: Dict[str, Dict[str, int]], labels: List[str]) -> Dict[str, float]:
    """Compute per-class precision from nested-dict confusion matrix."""
    precision = {}
    for label in labels:
        true_positive = confusion_matrix[label][label]
        predicted_positive = sum(confusion_matrix[true][label] for true in labels)
        precision[label] = (true_positive / predicted_positive) if predicted_positive > 0 else 0.0
    return precision


def recall_per_class(confusion_matrix: Dict[str, Dict[str, int]], labels: List[str]) -> Dict[str, float]:
    """Compute per-class recall from nested-dict confusion matrix."""
    recall = {}
    for label in labels:
        true_positive = confusion_matrix[label][label]
        actual_positive = sum(confusion_matrix[label][pred] for pred in labels)
        recall[label] = (true_positive / actual_positive) if actual_positive > 0 else 0.0
    return recall


def f1_per_class(
    precision: Dict[str, float],
    recall: Dict[str, float],
    labels: List[str],
) -> Dict[str, float]:
    """Compute per-class F1 score from per-class precision and recall."""
    f1_scores = {}
    for label in labels:
        p_val = precision.get(label, 0.0)
        r_val = recall.get(label, 0.0)
        denom = p_val + r_val
        f1_scores[label] = (2 * p_val * r_val / denom) if denom > 0 else 0.0
    return f1_scores
