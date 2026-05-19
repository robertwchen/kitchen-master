# Export for ChatGPT

## Repo tree Claude made

```text
kitchen-master/
├── .gitignore
├── README.md
├── requirements.txt
├── docs/
│   ├── plan.md
│   └── problem.md
├── experiments/
│   ├── configs/
│   │   └── sim_v1.yaml
│   ├── run_eval.py
│   └── run_sim.py
├── src/
│   ├── __init__.py
│   ├── baseline_detector.py
│   ├── config.py
│   ├── evaluate.py
│   └── sim_generator.py
└── tests/
    ├── __init__.py
    ├── test_detector.py
    ├── test_evaluate.py
    └── test_sim_generator.py
```

## README it wrote

````md
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
````

## File names it created for synthetic generation

```text
src/sim_generator.py
experiments/run_sim.py
experiments/configs/sim_v1.yaml
tests/test_sim_generator.py
src/config.py
src/evaluate.py
experiments/run_eval.py
```

## One example of the generated metadata CSV

```text
Note: the current repo does not generate a file literally named metadata.csv. The synthetic pipeline writes results to results/sim_v1/predictions.csv and results/sim_v1/metrics.csv, so the exported CSV example below uses predictions.csv as the closest generated artifact.

true,pred
legal,legal
fault,fault
legal,legal
uncertain,fault
legal,legal
uncertain,uncertain
fault,fault
uncertain,uncertain
legal,legal
fault,fault
uncertain,uncertain
uncertain,fault
uncertain,fault
uncertain,fault
uncertain,uncertain
```

## Also generated during verification

```text
n,precision_legal,recall_legal,precision_fault,recall_fault,precision_uncertain,recall_uncertain,uncertain_rate,false_fault_rate,missed_fault_rate
200,1.0,0.9,0.5319,1.0,0.918,0.56,0.305,0.0,0.0
```
