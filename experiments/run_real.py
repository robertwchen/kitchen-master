"""Run baseline detector on labeled real frames.

Expected annotations.csv format:
    frame_path,true_label,notes
    data/real/frames/clip01_042.jpg,legal,clear view
    data/real/frames/clip01_087.jpg,fault,foot over line
    data/real/frames/clip02_015.jpg,uncertain,shadow occlusion
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline_detector import predict
from src.evaluate import (
    compute_failure_analysis,
    compute_metrics,
    plot_confusion_matrix,
    save_confusion_matrix_csv,
    save_failure_analysis_csv,
    save_metrics_csv,
    save_predictions_csv,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def load_annotations(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def main(annotations_path: Path, results_dir: Path, detector_cfg: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_path.exists():
        logger.error(f"Annotations file not found: {annotations_path}")
        logger.error("Create data/real/annotations.csv following the template.")
        sys.exit(1)

    rows = load_annotations(annotations_path)
    if not rows:
        logger.error("annotations.csv is empty")
        sys.exit(1)

    y_true, y_pred, skipped = [], [], 0
    for row in rows:
        fpath = Path(row["frame_path"])
        if not fpath.exists():
            logger.warning(f"Frame not found, skipping: {fpath}")
            skipped += 1
            continue
        frame = cv2.imread(str(fpath))
        if frame is None:
            logger.warning(f"Could not read frame, skipping: {fpath}")
            skipped += 1
            continue
        y_true.append(row["true_label"])
        y_pred.append(predict(frame, detector_cfg))

    if skipped:
        logger.warning(f"{skipped} frame(s) skipped")
    if not y_true:
        logger.error("No valid frames found. Check frame paths in annotations.csv.")
        sys.exit(1)

    logger.info(f"Processed {len(y_true)} frames")
    metrics = compute_metrics(y_true, y_pred)

    logger.info(f"uncertain_rate={metrics['uncertain_rate']:.1%}")
    logger.info(f"false_fault_rate={metrics['false_fault_rate']:.1%}")
    logger.info(f"missed_fault_rate={metrics['missed_fault_rate']:.1%}")
    for label in ["legal", "fault", "uncertain"]:
        logger.info(
            f"{label}: precision={metrics[f'precision_{label}']:.3f}  recall={metrics[f'recall_{label}']:.3f}"
        )

    save_predictions_csv(y_true, y_pred, results_dir / "predictions.csv")
    save_metrics_csv(metrics, results_dir / "metrics.csv")
    save_confusion_matrix_csv(metrics, results_dir / "confusion_matrix.csv")
    plot_confusion_matrix(metrics, results_dir / "confusion_matrix.png")

    logger.info(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KitchenMaster real-data evaluation")
    parser.add_argument("--annotations", default="data/real/annotations.csv",
                        help="Path to annotations CSV")
    parser.add_argument("--results", default="results/real_v1",
                        help="Output directory for results")
    parser.add_argument("--fault-threshold-px", type=int, default=2)
    parser.add_argument("--uncertain-margin-px", type=int, default=8)
    parser.add_argument("--line-detection", default="hough", choices=["hough", "gradient"])
    args = parser.parse_args()

    detector_cfg = {
        "fault_threshold_px": args.fault_threshold_px,
        "uncertain_margin_px": args.uncertain_margin_px,
        "line_detection": args.line_detection,
    }
    main(Path(args.annotations), Path(args.results), detector_cfg)
