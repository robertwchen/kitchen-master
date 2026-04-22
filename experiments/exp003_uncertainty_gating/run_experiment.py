"""Scaffold runner for Experiment 003 (uncertainty gating)."""

from pathlib import Path
import json

import yaml


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("[INFO] exp003 scaffold only.")
    print("[INFO] No uncertainty model or thresholding is executed.")

    placeholder_path = Path(config.get("output_root", "results")) / "logs" / "exp003_uncertainty_gating_placeholder.json"
    placeholder_path.parent.mkdir(parents=True, exist_ok=True)
    placeholder_path.write_text(json.dumps({"status": "scaffold_only", "metrics": {}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
