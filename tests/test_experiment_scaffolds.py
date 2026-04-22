"""Tests for experiment scaffold utilities."""

from pathlib import Path

from experiments.exp001_synthetic_geometry.run_experiment import prepare_result_directories


def test_exp001_prepare_result_directories_creates_artifact_paths(tmp_path: Path) -> None:
    paths = prepare_result_directories(tmp_path, "exp001_synthetic_geometry")

    assert paths["run_root"].exists()
    assert paths["plots_dir"].exists()
    assert paths["config_path"].name == "config_snapshot.json"
    assert paths["metrics_path"].name == "metrics.json"
    assert paths["plot_manifest_path"].name == "plot_manifest.json"
