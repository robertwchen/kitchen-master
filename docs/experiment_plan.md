# Experiment Plan

## Experiment 1: Synthetic Geometry

- Objective: Validate geometry assumptions and label logic under controlled simulated layouts.
- Inputs: Deterministic scene parameters and line-foot placements.
- Outputs: Basic metric pipeline validation and visualization checks.

## Experiment 2: Real Controlled Baseline

- Objective: Establish initial performance on curated real captures with constrained camera setup.
- Inputs: Manually labeled frames following `docs/labeling_spec.md`.
- Outputs: Baseline confusion matrix and class-wise metrics.

## Experiment 3: Robustness

- Objective: Quantify sensitivity to practical perturbations (lighting, blur, angle, occlusion).
- Inputs: Stratified subsets or perturbation-defined slices.
- Outputs: Failure-mode-attributed metric shifts and error summaries.

## Experiment 4: Uncertainty Gating

- Objective: Evaluate abstention policy (`uncertain`) and selective prediction behavior.
- Inputs: Confidence heuristics or uncertainty signals from baseline methods.
- Outputs: Uncertain rate, selective accuracy, and coverage/accuracy tradeoff plots.
