# KitchenMaster Full Project History, Research Workflow, Planning Log, and Technical Documentation

## Project Overview

Project Name: KitchenMaster

Course: ECE 4501/6501 Sensors and Ubiquitous Computing

Student: Robert Chen

Core Project Goal:
Build and study a portable side-profile pickleball kitchen/NVZ foot-fault detection system using computer vision and synthetic plus real-world evaluation.

Original Proposal Goal:
A clip-on net-post device that detects kitchen/NVZ line contact during volleys and provides an immediate beep plus a log.

The proposal identified that:
- Kitchen foot faults are common sources of arguments in self-officiated pickleball.
- Existing professional systems are expensive and fixed-install.
- Portable public-court solutions do not exist at consumer scale.
- Robustness and confidence-aware outputs are important.

## Original Research Questions

### RQ1
Can a portable net-mounted sensor reliably detect NVZ line contact (foot on/over kitchen line) in real time?

### RQ2
How do mount location, lighting, occlusion, and court variation affect accuracy?

### RQ3
Can the system use confidence plus “uncertain” outputs to avoid wrong calls while maintaining low false alarms?

## Initial Project Concerns and Questions

The project started with uncertainty about:
- whether to use Codex or Codex Cloud
- whether a game engine should be used immediately
- whether ball bounce detection should come first
- how to structure a research workflow
- how to make the repo resume-quality
- how much should be automated by AI
- how to ensure the work remained scientifically honest

The user wanted:
- deep research
- realistic testing
- graphs and metrics
- reproducible experiments
- a strong GitHub repo
- a grad-style research workflow
- something defensible during presentation/Q&A

## Final Research Philosophy

One of the most important decisions made early was:

> If a phase fails, that is not failure. It is a research result.

The workflow became:
1. test assumptions
2. identify failure modes
3. redesign if necessary
4. document all results honestly

The project was intentionally framed as a staged research system rather than a fake polished AI demo.

## Final Division of Labor

### 1. User Responsibilities

The user owns:
- scientific judgment
- what counts as a violation
- labeling definitions
- evaluation decisions
- whether outputs are honest enough to present
- real-world sanity checks
- final presentation/report defense

### 2. ChatGPT Responsibilities

ChatGPT acts as:
- research lead
- project manager
- experiment designer
- planning coordinator

Responsibilities:
- convert proposal into research plan
- define milestones
- define metrics and baselines
- define experiments
- decide when to redesign
- determine when game engines are justified
- write Codex/Claude prompts
- help interpret results
- help write report/presentation language

### 3. Codex / Claude Code Responsibilities

AI coding agents act as implementation workers.

Responsibilities:
- create repo structure
- write scripts
- implement synthetic generators
- implement baselines
- create plots and metrics
- run tests
- organize experiments
- make clean commits

## Final Research Direction

The project was narrowed into a strong Version 1 research question:

### Version 1 Goal

Using a side-profile camera near the kitchen line, classify a player’s relevant foot state as:
- behind_line
- on_line
- over_line
- uncertain

## Why Ball Bounce Detection Was Deferred

Ball bounce detection was intentionally NOT chosen as the first step because:
1. A ball bouncing in the kitchen is not itself a fault.
2. Kitchen faults fundamentally depend on player position relative to the line.
3. Foot-vs-line geometry is the higher-value research risk.
4. Ball detection adds major complexity:
   - tiny object
   - motion blur
   - high speed
   - occlusion
   - timing ambiguity

## System Decomposition

### A. Court Geometry Detection
Where is the kitchen line in the image?

### B. Foot Localization
Where is the relevant foot relative to the kitchen line?

### C. Decision Logic
Should the system output:
- behind_line
- on_line
- over_line
- uncertain

## Why the Project Did NOT Start With a Game Engine

The decision was made NOT to start with a game engine because the first research risk was:

> “Can the geometry problem even be solved?”

The project intentionally started simple:
- simple synthetic side-profile scenes
- simple foot geometry
- interpretable baselines
- controlled experiments

## Final Planned Workflow

### Phase 0
Research definition and repo scaffold.

### Phase 1
Simple synthetic 2D geometry experiment.

### Phase 2
Controlled real-world baseline clips.

### Phase 3
Robustness experiments.

### Phase 4
Confidence and uncertainty gating.

### Phase 5
Optional game-engine/3D realism.

### Phase 6
Final report and presentation.

## Repo Structure

```text
kitchen-master/
  README.md
  AGENTS.md
  requirements.txt
  .gitignore

  docs/
    project_thesis.md
    labeling_spec.md
    scenarios.md
    metrics.md
    failure_modes.md
    experiment_plan.md

  data/
    synthetic/
    real/
    labels/

  sim/
    scene_generation/
    assets/

  src/
    geometry/
    detection/
    decision/
    evaluation/
    utils/

  experiments/
    exp001_synthetic_geometry/
    exp002_real_controlled_baseline/
    exp003_uncertainty_gating/

  results/
    plots/
    tables/
    logs/

  tests/
```

## Labels

### behind_line
Foot fully behind kitchen line.

### on_line
Any visible part of shoe overlaps kitchen line.

### over_line
Any visible part of shoe extends into kitchen.

### uncertain
Insufficient evidence due to:
- blur
- occlusion
- bad angle
- poor line visibility
- ambiguous overlap

## Initial Four Scenarios

### S1 Clear Legal
Expected:
behind_line

### S2 Clear Fault On Line
Expected:
on_line

### S3 Clear Fault Over Line
Expected:
over_line

### S4 Ambiguous
Expected:
uncertain

## Metrics

### Primary Metrics
- accuracy
- precision
- recall
- F1
- confusion matrix

### Most Important Metrics
- fault precision
- false positive rate
- uncertain rate
- selective accuracy

## Failure Modes

1. faded line paint
2. low contrast line
3. shoe blends with court
4. net/post blocks line
5. imperfect side angle
6. foot off-frame
7. motion blur
8. shadows
9. glare
10. multiple players

## Codex Cloud Decisions

Initial settings:

```text
Container: universal
Internet: OFF
Setup script: empty
Simultaneous agents: 1x
```

## Final Tool Workflow

### Claude Code
Used for:
- fast implementation
- synthetic generator
- baseline code
- quick iteration

### Codex Cloud
Used for:
- structured runs
- clean experiment execution
- reproducibility
- repo polish

### ChatGPT
Used for:
- research planning
- workflow decisions
- experiment design
- interpretation

## Phase 0 Deliverables

Key generated files:
- README.md
- AGENTS.md
- docs/*
- experiments/*
- metrics utilities
- plotting utilities
- tests

## Main Phase 0 Prompt

```text
IMPORTANT:
Only scaffold the research repository.
Do NOT implement detection models yet.
Do NOT invent fake data.
Do NOT download datasets.
Create structure, docs, experiment scaffolding, and utilities only.
```

## Phase 1 Goals

Build:
- synthetic dataset
- baseline classifier
- metrics pipeline
- plots
- first real experimental evidence

## Planned Synthetic Parameters

- image size
- line thickness
- line position
- foot position
- foot size
- foot rotation
- blur
- brightness
- occlusion
- visibility degradation
- background variation
- random seed

## Planned Outputs

```text
results/exp001/
  metrics.json
  confusion_matrix.png
  class_metrics.png
  sample_predictions/
```

## Planned Baseline

Important design decision:
NO deep learning initially.

Reason:
- easier to interpret
- easier to debug
- more honest research baseline

## Final Report Structure

1. Introduction
2. Research Questions
3. Scope
4. Method
5. Implementation
6. Experiments
7. Results
8. Discussion
9. Limitations
10. Future Work
11. Conclusion

## Final Recommended Framing

KitchenMaster is not claiming to fully automate pickleball officiating. Instead, it studies the first core sensing problem behind a portable kitchen-fault assistant: whether a side-profile camera can reliably classify foot position relative to the NVZ boundary and abstain when the evidence is uncertain.
