"""Evaluation metrics and output utilities."""

import csv
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

LABELS = ["legal", "fault", "uncertain"]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    n = len(y_true)
    label_idx = {l: i for i, l in enumerate(LABELS)}

    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            cm[label_idx[t], label_idx[p]] += 1

    metrics: dict = {"n": n, "confusion_matrix": cm.tolist()}

    for i, label in enumerate(LABELS):
        tp = cm[i, i]
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        metrics[f"precision_{label}"] = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        metrics[f"recall_{label}"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0

    metrics["uncertain_rate"] = round(y_pred.count("uncertain") / n, 4)
    metrics["false_fault_rate"] = round(
        sum(1 for t, p in zip(y_true, y_pred) if t == "legal" and p == "fault") / n, 4
    )
    metrics["missed_fault_rate"] = round(
        sum(1 for t, p in zip(y_true, y_pred) if t == "fault" and p == "legal") / n, 4
    )
    return metrics


def save_metrics_csv(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)
    logger.info(f"Saved metrics to {path}")


def save_predictions_csv(y_true: list[str], y_pred: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["true", "pred"])
        writer.writeheader()
        writer.writerows({"true": t, "pred": p} for t, p in zip(y_true, y_pred))
    logger.info(f"Saved predictions to {path}")


def plot_confusion_matrix(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.array(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {path}")
