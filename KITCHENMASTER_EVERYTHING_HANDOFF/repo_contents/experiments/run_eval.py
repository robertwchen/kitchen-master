"""Re-run evaluation on a saved predictions.csv from any prior experiment."""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import compute_metrics, plot_confusion_matrix, save_metrics_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(path: Path) -> tuple[list[str], list[str]]:
    y_true, y_pred = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            y_true.append(row["true"])
            y_pred.append(row["pred"])
    return y_true, y_pred


def main(results_dir: Path) -> None:
    pred_path = results_dir / "predictions.csv"
    if not pred_path.exists():
        logger.error(f"predictions.csv not found in {results_dir}")
        logger.error("Run run_sim.py or label real data first.")
        sys.exit(1)

    y_true, y_pred = load_predictions(pred_path)
    metrics = compute_metrics(y_true, y_pred)

    save_metrics_csv(metrics, results_dir / "metrics.csv")
    plot_confusion_matrix(metrics, results_dir / "confusion_matrix.png")

    logger.info(f"uncertain_rate={metrics['uncertain_rate']:.1%}")
    logger.info(f"false_fault_rate={metrics['false_fault_rate']:.1%}")
    logger.info(f"missed_fault_rate={metrics['missed_fault_rate']:.1%}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate from saved predictions")
    parser.add_argument("--results", required=True, help="Path to results directory")
    args = parser.parse_args()
    main(Path(args.results))
