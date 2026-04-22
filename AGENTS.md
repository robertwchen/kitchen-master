# KitchenMaster Agent Guidelines

## Core Engineering Principles

- Keep code simple, modular, and easy to inspect.
- Prefer interpretable baselines before machine learning methods.
- Write readable, minimal Python and avoid unnecessary abstractions.
- Avoid large frameworks unless they are clearly necessary.

## Experiment Discipline

- Every experiment must save:
  - configuration
  - metrics
  - plots
- Use the `results/` directory consistently for all outputs.

## Data and Result Integrity

- Never generate fake data.
- Never hardcode results.
- Add `TODO` comments instead of guessing when requirements are unclear.

## Repository Intent

This is a research repository. Prioritize clarity, traceability, and reproducibility over feature breadth.
