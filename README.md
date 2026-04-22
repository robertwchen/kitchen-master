# KitchenMaster

KitchenMaster is a research repository for studying whether a portable side-profile camera mounted near a pickleball net can detect kitchen (non-volley zone, NVZ) foot position relative to the kitchen line.

## Version 1 Task Definition

Version 1 is a line-relative foot-state classification problem with four classes:
- `behind_line`
- `on_line`
- `over_line`
- `uncertain`

The focus is on interpretable, reproducible research scaffolding rather than production deployment.

## Research Phases

- **Phase 0:** Definitions, repository scaffolding, and evaluation foundations
- **Phase 1:** Synthetic geometry and controlled scene assumptions
- **Phase 2:** Real controlled baseline experiments
- **Phase 3:** Robustness analysis under practical variation
- **Phase 4:** Uncertainty handling and selective decision policies

## Repository Structure (High Level)

- `docs/` — thesis, labeling standards, scenarios, metrics, failure modes, and experiment plan
- `data/` — reserved structure for synthetic, real, and labels (no data included)
- `sim/` — scene-generation and simulation assets scaffolding
- `src/` — geometry, detection, decision, evaluation, and utility modules
- `experiments/` — experiment-specific configs and runnable scaffolds for:
  - `exp001_synthetic_geometry`
  - `exp002_real_controlled_baseline`
  - `exp003_uncertainty_gating`
- `results/` — metrics, plots, tables, and logs outputs
- `tests/` — lightweight validation of utilities and scaffolding behavior

## Scope Note

This repository is intentionally research-first and minimal. It does **not** include model training pipelines, production APIs, or dataset downloads at this stage.
