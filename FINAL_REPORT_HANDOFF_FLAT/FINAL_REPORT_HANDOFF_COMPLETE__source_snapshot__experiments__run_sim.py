"""Run the full synthetic pipeline: generate → detect → evaluate → save."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline_detector import predict
from src.config import get_default_config, load_config
from src.evaluate import (
    compute_failure_analysis,
    compute_metrics,
    plot_confusion_matrix,
    plot_qualitative_overlays,
    save_confusion_matrix_csv,
    save_failure_analysis_csv,
    save_metrics_csv,
    save_predictions_csv,
)
from src.sim_generator import generate_dataset, save_metadata_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def main(cfg: dict) -> None:
    run_name = cfg["run_name"]
    results_dir = Path(cfg["output"]["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    samples = generate_dataset(cfg["sim"])
    metas = [meta for _, meta in samples]

    y_true = [meta.ground_truth_label for meta in metas]
    y_pred = [predict(frame, cfg["detector"]) for frame, _ in samples]

    metrics = compute_metrics(y_true, y_pred)
    failure_rows = compute_failure_analysis(metas, y_pred)

    logger.info(f"n={metrics['n']}")
    logger.info(f"uncertain_rate={metrics['uncertain_rate']:.1%}")
    logger.info(f"false_fault_rate={metrics['false_fault_rate']:.1%}")
    logger.info(f"missed_fault_rate={metrics['missed_fault_rate']:.1%}")
    for label in ["legal", "fault", "uncertain"]:
        logger.info(
            f"{label}: precision={metrics[f'precision_{label}']:.3f}  recall={metrics[f'recall_{label}']:.3f}"
        )

    save_metadata_csv(metas, results_dir / "metadata.csv")
    save_predictions_csv(y_true, y_pred, results_dir / "predictions.csv")
    save_metrics_csv(metrics, results_dir / "metrics.csv")
    save_confusion_matrix_csv(metrics, results_dir / "confusion_matrix.csv")
    save_failure_analysis_csv(failure_rows, results_dir / "failure_analysis.csv")

    if cfg["output"]["save_plots"]:
        plot_confusion_matrix(metrics, results_dir / "confusion_matrix.png")
    if cfg["output"].get("save_overlays", True):
        plot_qualitative_overlays(samples, y_pred, results_dir, n_per_group=3)

    logger.info(f"Done. Results in {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KitchenMaster synthetic pipeline")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else get_default_config()
    if not args.config:
        logger.info("No config provided — using defaults")
    main(cfg)
