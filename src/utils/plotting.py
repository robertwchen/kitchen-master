"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix_plot(
    confusion_matrix: Dict[str, Dict[str, int]],
    labels: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Save a confusion matrix heatmap to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.array([[confusion_matrix[t][p] for p in labels] for t in labels], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_class_metrics_bar_chart(
    metrics: Dict[str, Dict[str, float]],
    labels: List[str],
    output_path: str,
    title: str = "Class Metrics",
) -> None:
    """Save a grouped bar chart for per-class metrics."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metric_names = list(metrics.keys())
    x_pos = np.arange(len(labels))
    width = 0.8 / max(1, len(metric_names))

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, metric_name in enumerate(metric_names):
        values = [metrics[metric_name].get(label, 0.0) for label in labels]
        ax.bar(x_pos + idx * width, values, width=width, label=metric_name)

    ax.set_xticks(x_pos + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
