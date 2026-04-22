"""Scaffold runner for Experiment 002 (real controlled baseline)."""

from pathlib import Path
import json

import yaml


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("[INFO] exp002 scaffold only.")
    print("[INFO] No real data processing is executed.")

    placeholder_path = Path(config.get("output_root", "results")) / "logs" / "exp002_real_controlled_baseline_placeholder.json"
    placeholder_path.parent.mkdir(parents=True, exist_ok=True)
    placeholder_path.write_text(json.dumps({"status": "scaffold_only", "metrics": {}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
