# KitchenMaster

Research prototype for pickleball NVZ foot-fault detection from a fixed side-view camera.

## Research Questions

- **RQ1**: Can a fixed side-view camera detect NVZ line contact in controlled conditions?
- **RQ2**: How sensitive is detection to viewpoint, blur, occlusion, and foot-line distance?
- **RQ3**: Can an uncertain output reduce wrong calls in ambiguous cases?

## Output Labels

| Label | Meaning |
|-------|---------|
| `legal` | Foot clearly behind the line |
| `fault` | Foot touches or crosses the line |
| `uncertain` | Cannot be determined reliably |

## Repo Structure

```
docs/                  problem definition and research plan
data/sim/              generated synthetic frames and labels
data/real/             collected real clips and labels
src/                   core library: config, generation, detection, evaluation
experiments/           runnable experiment scripts and YAML configs
results/               saved metrics CSVs and plots (large files gitignored)
tests/                 unit tests
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run synthetic pipeline (generates data, runs detector, saves results)
python experiments/run_sim.py

# Run with a specific config
python experiments/run_sim.py --config experiments/configs/sim_v1.yaml

# Re-evaluate from saved predictions
python experiments/run_eval.py --results results/sim_v1/

# Run tests
pytest tests/
```

## Reproducibility

All experiment outputs (metrics, confusion matrices) are saved under `results/<run_name>/`. Configs and code are version-controlled. Raw data is gitignored but generation is seeded and deterministic.
