"""Evaluation metrics and output utilities."""

import csv
import logging
from pathlib import Path

import cv2
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


def save_confusion_matrix_csv(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = metrics["confusion_matrix"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label\\pred_label"] + LABELS)
        for i, label in enumerate(LABELS):
            writer.writerow([label] + cm[i])
    logger.info(f"Saved confusion matrix CSV to {path}")


def save_predictions_csv(y_true: list[str], y_pred: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["true", "pred"])
        writer.writeheader()
        writer.writerows({"true": t, "pred": p} for t, p in zip(y_true, y_pred))
    logger.info(f"Saved predictions to {path}")


def compute_failure_analysis(metas: list, y_pred: list[str]) -> list[dict]:
    """Group prediction outcomes by scenario, occlusion, blur, and distance bucket."""

    def _dist_bucket(d: int) -> str:
        if d < -10:
            return "< -10"
        if d < -3:
            return "-10 to -3"
        if d < 3:
            return "-3 to +3"
        if d < 10:
            return "+3 to +10"
        if d < 20:
            return "+10 to +20"
        return "> +20"

    groups: dict[tuple, list] = {}
    for meta, pred in zip(metas, y_pred):
        bucket = _dist_bucket(meta.signed_distance_px)
        key = (meta.scenario_type, meta.occlusion_flag, meta.blur_level, bucket)
        groups.setdefault(key, []).append((meta, pred))

    rows = []
    for (scenario, occluded, blur, bucket), items in sorted(groups.items()):
        n = len(items)
        correct = sum(1 for m, p in items if p == m.ground_truth_label)
        rows.append({
            "scenario_type": scenario,
            "occlusion_flag": occluded,
            "blur_level": blur,
            "distance_bucket_px": bucket,
            "n_samples": n,
            "n_correct": correct,
            "accuracy": round(correct / n, 4),
            "predicted_legal": sum(1 for _, p in items if p == "legal"),
            "predicted_fault": sum(1 for _, p in items if p == "fault"),
            "predicted_uncertain": sum(1 for _, p in items if p == "uncertain"),
        })
    return rows


def save_failure_analysis_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved failure analysis to {path} ({len(rows)} groups)")


def plot_qualitative_overlays(
    samples: list[tuple],
    y_pred: list[str],
    results_dir: Path,
    n_per_group: int = 3,
) -> None:
    """
    Save annotated overlay images grouped by (scenario_type, correct/wrong).

    Each overlay draws:
      - Blue line  = ground-truth kitchen line (from frame)
      - Green line = detected kitchen line
      - Orange line = ground-truth foot bottom
      - Red line   = detected foot bottom
    """
    from src.baseline_detector import detect_foot_bottom, detect_line_y

    overlays_dir = results_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[tuple, list] = {}
    for (frame, meta), pred in zip(samples, y_pred):
        correct = pred == meta.ground_truth_label
        key = (meta.scenario_type, "correct" if correct else "wrong")
        buckets.setdefault(key, []).append((frame, meta, pred))

    saved = 0
    for (scenario, outcome), items in sorted(buckets.items()):
        for idx, (frame, meta, pred) in enumerate(items[:n_per_group]):
            det_line_y = detect_line_y(frame)
            det_foot_bottom = detect_foot_bottom(frame)

            overlay = frame.copy()
            W = overlay.shape[1]
            gt_foot_bottom = meta.foot_y + meta.foot_height

            # Ground-truth lines (blue = kitchen line already in frame; draw foot bottom in orange)
            cv2.line(overlay, (0, gt_foot_bottom), (W, gt_foot_bottom), (0, 140, 255), 1)

            # Detected kitchen line (green)
            if det_line_y is not None:
                cv2.line(overlay, (0, det_line_y), (W, det_line_y), (0, 210, 0), 1)

            # Detected foot bottom (red)
            if det_foot_bottom is not None:
                cv2.line(overlay, (0, det_foot_bottom), (W, det_foot_bottom), (0, 0, 220), 1)

            correct_str = "OK" if pred == meta.ground_truth_label else "FAIL"
            cv2.putText(overlay, f"GT:{meta.ground_truth_label} PRED:{pred} [{correct_str}]",
                        (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)
            cv2.putText(overlay, f"dist={meta.signed_distance_px}px blur={meta.blur_level}",
                        (3, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)

            fname = overlays_dir / f"{scenario}_{outcome}_{idx:02d}.png"
            cv2.imwrite(str(fname), overlay)
            saved += 1

    logger.info(f"Saved {saved} overlay images to {overlays_dir}/")


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
