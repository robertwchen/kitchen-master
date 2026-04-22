"""Scaffold runner for Experiment 001 (synthetic geometry)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    """Load experiment configuration from a YAML file."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare_result_directories(output_root: Path, experiment_name: str) -> dict:
    """Create a timestamped run directory and standard artifact paths."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = output_root / "logs" / experiment_name / timestamp
    plots_dir = output_root / "plots" / experiment_name / timestamp

    run_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_root": run_root,
        "plots_dir": plots_dir,
        "config_path": run_root / "config_snapshot.json",
        "metrics_path": run_root / "metrics.json",
        "log_path": run_root / "run.log",
        "plot_manifest_path": run_root / "plot_manifest.json",
    }


def save_placeholder_metrics(metrics_path: Path) -> None:
    """Save an empty metric payload without fabricated values."""
    payload = {
        "status": "scaffold_only",
        "metrics": {},
        "notes": "No model outputs or fabricated numbers are stored in this stage.",
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_placeholder_plot_manifest(plot_manifest_path: Path, plots_dir: Path) -> None:
    """Save an empty plot manifest for traceability during scaffold runs."""
    payload = {
        "status": "scaffold_only",
        "plots_dir": str(plots_dir),
        "plots": [],
        "notes": "No plots are generated during scaffold-only runs.",
    }
    plot_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)

    output_root = Path(config.get("output_root", "results"))
    experiment_name = config.get("experiment_name", "exp001_synthetic_geometry")
    paths = prepare_result_directories(output_root, experiment_name)

    paths["config_path"].write_text(json.dumps(config, indent=2), encoding="utf-8")

    log_lines = [
        "[INFO] Starting experiment scaffold.",
        "[INFO] Config loaded successfully.",
        "[INFO] No detection model executed (scaffold-only run).",
        "[INFO] Saving placeholder metrics structure.",
        "[INFO] Recording empty plot manifest for this run.",
        "[INFO] Completed scaffold run.",
    ]
    paths["log_path"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    for line in log_lines:
        print(line)

    save_placeholder_metrics(paths["metrics_path"])
    save_placeholder_plot_manifest(paths["plot_manifest_path"], paths["plots_dir"])


if __name__ == "__main__":
    main()
