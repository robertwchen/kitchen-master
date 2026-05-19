# KitchenMaster Technical Summary

## Purpose
`KitchenMaster` is a research prototype for detecting pickleball kitchen / non-volley zone (NVZ) foot faults from a single fixed side-view camera.

The project tries to answer three research questions:

1. Can a side-view camera detect whether a foot stayed behind the kitchen line or crossed it?
2. How sensitive is that decision to blur, occlusion, camera angle, and near-line ambiguity?
3. Is it better to output `uncertain` than force a wrong `legal` or `fault` label?

The core pipeline is:

1. Estimate court geometry in the image.
2. Track the ball and infer likely event timing.
3. Localize the relevant foot.
4. Compute signed distance from foot to NVZ boundary.
5. Output `legal`, `fault`, or `uncertain`.
6. Support human review and overrides for hard cases.

## Top-Level Repo Structure

### `src/`
Core Python modules for geometry, registration, tracking, localization, and evaluation.

### `experiments/`
Runnable entrypoints plus YAML configs.

### `scripts/`
Helper tools for frame extraction and manual annotation.

### `docs/`
Problem statement and project plan.

### `tests/`
Unit tests.

### `data/`
Raw videos, frames, annotations, and other inputs. Large assets are mostly gitignored.

### `results/`
Generated CSVs, reports, images, and videos from experiments.

### `misc/`
Ad hoc analysis artifacts and export files.

### `models/`
Model files such as ONNX pose weights.

## System Evolution

### Phase 0: Synthetic baseline
Goal: prove the problem framing, dataset generation, evaluation logic, and baseline detector.

Main deliverables:

- synthetic scene generation
- baseline line + foot detector
- metrics and failure analysis
- tests

### Phase 1: Real court registration
Goal: estimate where the actual kitchen boundaries are in real video.

Versions:

- `court_reg_v1`: simpler line-based approach
- `court_reg_v2`: ORB-based stabilization and line warping
- `court_reg_v3`: current anchor-point court model plus ORB-based stabilization

### Phase 2: End-to-end presentation demo
Goal: connect court registration, ball tracking, event inference, foot localization, and final foot-fault labeling into a reviewable demo pipeline.

This phase is explicitly human-in-the-loop. The code is designed to stop after generating review artifacts, then rerun after user corrections.

## Core Python Modules

## Configuration

### `src/config.py`
Loads YAML configs and provides default config values for early experiments.

## Synthetic / baseline pipeline

### `src/sim_generator.py`
Generates synthetic foot-fault scenes and per-sample metadata.

Typical metadata includes:

- line position
- foot location or box
- true label
- signed distance
- scenario type
- blur / occlusion flags
- random seed

### `src/baseline_detector.py`
Implements an interpretable classical CV baseline.

Core logic:

- detect a court line
- detect the foot with simple color / geometry logic
- compute foot-to-line gap
- classify as `legal`, `fault`, or `uncertain`

### `src/evaluate.py`
Computes metrics and failure analysis.

Likely outputs include:

- confusion matrix
- precision / recall
- false fault rate
- missed fault rate
- uncertain rate
- grouped error analysis

## Court geometry and registration

### `src/court_registration.py`
Older registration logic built around explicit line representations.

Important concepts:

- `LineModel`
- signed distance from a point to a line
- line stability measurements
- v1-style line registration

### `src/court_model.py`
Defines the anchor-based court geometry abstraction used by the modern pipeline.

Derives:

- near kitchen line
- far kitchen line
- left NVZ boundary
- right NVZ boundary
- legal side polygons

Supports warping the court model under an estimated transform.

### `src/stabilizer.py`
Handles per-frame motion estimation.

Main ideas:

- ORB feature extraction
- descriptor matching
- RANSAC transform estimation
- sanity checks and ROI masking

This is a key part of `court_reg_v2` and `court_reg_v3`.

### `src/viz.py`
Visualization utilities for drawing court models, debug overlays, and output videos.

## Ball and event reasoning

### `src/ball_detector.py`
Optional learned detector wrapper based on Ultralytics.

Typical role:

- generate candidate ball detections
- optionally use tiled inference
- filter for ball-like classes

### `src/ball_tracker.py`
Tracks the ball through a video sequence.

Main ideas:

- motion differencing
- HSV constraints
- optional Ultralytics proposals
- temporal linking
- trajectory smoothing
- CSV and overlay export

Typical outputs:

- `ball_tracking.csv`
- `ball_overlay.mp4`
- sampled debug frames

### `src/volley_classifier.py`
Uses the tracked ball trajectory to infer bounce candidates and event timing.

Main ideas:

- smooth the trajectory
- inspect motion before and after candidate frames
- detect bounce-like behavior
- optionally classify events when hit frames are known
- export review montages

Typical outputs:

- `candidates.csv`
- `events.csv`
- montage images

### `src/event_detector.py`
Placeholder module. It is not the current event logic path and is effectively superseded by `src/volley_classifier.py`.

## Foot localization and final decision

### `src/foot_localizer.py`
Localizes the relevant foot for event frames.

Supported modes include:

- `background_subtraction`
- `roi_threshold`
- `manual_point`
- `event_hybrid`

The hybrid path combines multiple cues, including ROI constraints and optional ONNX pose inference through OpenCV DNN.

### `src/foot_fault_pipeline.py`
Final decision stage.

For each event:

1. Load court geometry for that frame.
2. Choose the relevant NVZ boundary.
3. Localize the foot.
4. Compute signed distance from foot to line.
5. Classify:
   - `legal_volley`
   - `foot_fault_volley`
   - `uncertain`

It also handles:

- manual line overrides
- active-side inference
- CSV export
- annotated review frames

## Experiment Runners

### `experiments/run_sim.py`
Runs the synthetic end-to-end pipeline:

- generate data
- run baseline detector
- evaluate results
- save outputs

### `experiments/run_eval.py`
Recompute evaluation metrics from saved predictions.

### `experiments/run_real.py`
Run the baseline detector on labeled real frames.

### `experiments/run_court_registration.py`
Phase 1 v1 registration pipeline.

### `experiments/run_court_registration_v2.py`
Phase 1 v2 registration pipeline using ORB-based stabilization.

### `experiments/run_court_registration_v3.py`
Current registration pipeline:

- anchor-point court model
- ORB-based stabilization
- per-frame transforms
- validation and comparison exports

### `experiments/run_demo_pipeline.py`
Main end-to-end demo orchestrator.

It wires together:

- precomputed registration
- ball tracking
- bounce / volley inference
- foot localization
- foot-fault classification
- review / override workflow

## Helper Scripts

### `scripts/extract_frames.py`
Extract frames from video.

### `scripts/annotate_reference.py`
Interactive tool for older line annotation workflow.

### `scripts/annotate_anchors.py`
Interactive tool for anchor-point court geometry annotation.

### `scripts/annotate_reprojection_anchors.py`
Annotate sampled frames for reprojection validation.

## YAML Configs

### `experiments/configs/sim_v1.yaml`
Controls the synthetic experiment.

Typical knobs:

- run name
- synthetic data generation parameters
- detector thresholds
- output options

### `experiments/configs/court_reg_v1.yaml`
Controls the first registration experiment.

Typical knobs:

- video path
- annotation path
- refinement settings
- output settings

### `experiments/configs/court_reg_v2.yaml`
Controls ORB-based line registration.

Typical knobs:

- ORB / matching parameters
- transform settings
- refinement options
- output settings

### `experiments/configs/court_reg_v3.yaml`
Controls current anchor-based court registration.

Typical knobs:

- anchor annotations
- ORB stabilizer settings
- ROI and transform sanity settings
- validation settings
- comparison exports
- output settings

### `experiments/configs/demo_pipeline.yaml`
Controls the end-to-end presentation demo.

Important sections:

- `video`
- `registration`
- `ball_tracking`
- `volley_classification`
- `foot_localizer`
- `foot_fault`
- `pipeline`
- `output`

This is the best single config file to understand how the current system is wired.

## End-to-End Data Flow

### Step 1: Offline annotation
Manually annotate court anchors in a reference frame.

### Step 2: Court registration
`run_court_registration_v3.py` uses anchor annotations plus ORB-based stabilization to estimate court geometry for every frame.

Outputs include:

- per-frame registration CSV
- debug frames
- overlay video
- summary JSON

### Step 3: Demo pipeline loads registration
`run_demo_pipeline.py` usually reads the registration CSV rather than recomputing registration inside the demo.

### Step 4: Ball tracking
The pipeline tracks the ball using classical CV and optionally Ultralytics proposals.

Outputs include:

- `ball_tracking.csv`
- `ball_overlay.mp4`
- debug frames

### Step 5: Bounce / volley reasoning
Tracked ball positions are converted into bounce candidates and event timing hypotheses.

### Step 6: Active-side selection
The system infers which player side is active, often from nearby ball positions.

### Step 7: Foot localization
The pipeline estimates the relevant foot contact point for each event frame.

### Step 8: Signed-distance decision
The foot point is measured relative to the selected NVZ boundary.

Decision rule:

- clearly behind line -> legal
- clearly inside kitchen -> fault
- near threshold or ambiguous -> uncertain

### Step 9: Human review loop
In `auto_review`, the pipeline exports checkpoints and writes `review_pending.json`, then stops.

The reviewer can override:

- line geometry
- ball points
- bounce labels
- foot points
- active side
- final event label

### Step 10: Final outputs
In `apply_overrides`, the pipeline uses the edited review file to generate corrected outputs and final summaries.

## Review Architecture

### `auto_review`
Runs the pipeline, exports artifacts, writes `review_pending.json`, and stops.

### `apply_overrides`
Loads `review_approved.json`, applies corrections, and produces final outputs.

This makes the current system a research-demo workflow rather than a fully autonomous referee.

## Outputs and Results Structure

### Synthetic outputs
Usually under `results/sim_v1/`:

- metadata CSV
- predictions CSV
- metrics CSV
- confusion matrix
- failure analysis
- qualitative overlays

### Registration outputs
Usually under `results/real_baseline/court_reg_v*/`:

- transform or line CSV
- `summary_report.json`
- debug frames
- overlay video
- comparison outputs

### Demo outputs
Usually under `results/presentation_demo/<run_name>/`:

- `review/`
- `ball_tracking/`
- `volley_events/`
- `foot_faults/`
- `foot_faults_final/`
- `summary/`

Important files include:

- `review_pending.json`
- `review_approved.json`
- `ball_tracking.csv`
- `candidates.csv`
- `events.csv`
- `foot_fault_events.csv`
- `summary.json`
- `pipeline_summary.json`
- `demo_summary.mp4`

## Dependencies / Tech Stack

Main dependencies from `requirements.txt`:

- `numpy`
- `pandas`
- `matplotlib`
- `opencv-python`
- `pyyaml`
- `scikit-learn`
- `pytest`
- `ultralytics`

Practical interpretation:

- Python for orchestration
- OpenCV for image processing, video I/O, and ONNX inference
- NumPy / Pandas for numeric and tabular work
- Matplotlib for plots
- scikit-learn for evaluation utilities
- PyYAML for config-driven experiments
- Ultralytics YOLO for optional learned ball detection

## Implemented vs Placeholder

### Implemented

- synthetic dataset generation
- baseline detector
- evaluation and failure analysis
- line and court geometry abstractions
- court registration v1 / v2 / v3
- ORB stabilization
- visualization tools
- ball tracking
- bounce / volley heuristics
- foot localization
- foot-fault decision logic
- review / override workflow
- configs and experiment runners
- tests

### Placeholder or still experimental

- `src/event_detector.py` is not the main event path
- ball tracking is still heuristic / tuned
- volley inference is not fully robust
- foot localization still needs human review in difficult frames
- final polished outputs depend on the review loop
- this is not yet a fully automatic production-grade referee system

## Best Files To Read First

1. `README.md`
2. `docs/problem.md`
3. `experiments/configs/demo_pipeline.yaml`
4. `experiments/run_demo_pipeline.py`
5. `experiments/run_court_registration_v3.py`
6. `src/foot_fault_pipeline.py`
7. `src/ball_tracker.py`
8. `src/volley_classifier.py`
9. `src/foot_localizer.py`
10. `src/court_model.py`
11. `src/stabilizer.py`

## Current Results Snapshot

These are the strongest current numbers visible in the repo.

### Synthetic baseline (`README.md`)

- `sim_v1` used `200` frames with seed `42`
- false fault rate: `0.0%`
- missed fault rate: `0.0%`
- uncertain rate: `27.0%`
- legal precision / recall: `1.000 / 0.940`
- fault precision / recall: `0.505 / 1.000`

Interpretation:

- the synthetic baseline is conservative
- it catches faults well in the toy setup
- but many fault predictions are not precise, and uncertainty is used heavily

### Court registration v3 (`results/real_baseline/court_reg_v3/summary_report.json`)

- video: `pickle_vid_1_trimmed_from_8s.mp4`
- resolution: `1920x1080`
- fps: `59.943`
- total frames: `2055`
- duration: `34.28 s`
- method: `anchor-point court model + ORB post_translation`
- registration success: `2055 / 2055`
- fallbacks: `0`
- fallback rate: `0.0`
- sampled validation frames: `60`
- left boundary edge strength mean: `12.14`
- right boundary edge strength mean: `7.31`
- transform translation mean: `25.03 px`

Important comparison in the same report:

- `post_translation` fixed: `2055 ok`, `0 fallback`
- `affine` fixed: `2036 ok`, `19 fallback`

Interpretation:

- the current registration approach is the strongest technical result in the repo
- it is more stable than the compared affine configuration on this clip

### Demo pipeline snapshot (`results/presentation_demo/demo_v1_active_side/...`)

- pipeline mode: `auto_review`
- ball tracking detection rate: `35.7%`
- bounce candidates: none detected in the pending review file
- final event count: `3`
- labels:
  - `1` `foot_fault_volley`
  - `2` `uncertain`
- fault threshold: `5 px`
- uncertain margin: `15 px`
- foot localization mode: `event_hybrid`

Interpretation:

- the full pipeline is wired and producing reviewable event outputs
- but the end-to-end automatic path is still fragile, especially at the ball/event stage
- this is best presented as a working research prototype with human validation, not a finished automatic system

## Result Statistics A Professor Will Likely Ask About

Be ready to answer these with actual numbers where possible.

### Dataset / scope

- How many synthetic samples did you generate?
- How many real clips did you process?
- What are the video resolution, fps, and duration?
- How many event frames are you evaluating in the demo?

### Registration quality

- What percent of frames were successfully registered?
- How many fallback frames were needed?
- How did v3 compare against earlier registration methods?
- How did you validate that the projected line stayed correct over time?

### Detection / classification quality

- What are precision and recall for `legal`, `fault`, and `uncertain`?
- What is the false fault rate?
- What is the missed fault rate?
- How often does the system say `uncertain`?
- How close to the line do predictions become unstable?

### Demo pipeline quality

- What percent of frames had a valid ball detection?
- How many bounce candidates were found?
- How many final events were labeled as fault, legal, or uncertain?
- How often did you need manual overrides?
- Which stage fails most often: registration, ball tracking, event timing, or foot localization?

### Robustness questions

- What happens under motion blur?
- What happens under occlusion?
- What happens if the ball is missed?
- What happens if the wrong player side is inferred?
- How sensitive are results to thresholds like `fault_threshold_px` and `uncertain_margin_px`?

## Common Professor Questions

### “Why not just detect the foot directly?”
Because the foot location alone is meaningless unless the NVZ boundary is accurately registered in the same frame. Geometry is the foundation.

### “Why is `uncertain` useful?”
Because a wrong confident fault call is worse than abstaining. The project explicitly studies whether uncertainty reduces false calls.

### “What is your strongest result?”
Court registration v3 on the real clip: `2055/2055` successful registrations with `0` fallbacks.

### “What is your weakest stage right now?”
The ball/event stage in the presentation demo. The current `auto_review` snapshot shows only `35.7%` ball detection rate and no bounce candidates in that run.

### “Is this fully automatic?”
Not yet. The current presentation demo is intentionally human-in-the-loop with a review JSON and override workflow.

### “Why should I trust the final label?”
Because the decision is interpretable: court boundary + selected player side + localized foot point + signed distance + explicit threshold / uncertainty margin.

### “What would you improve next?”

- improve ball detector or train a dedicated pickleball detector
- improve event timing logic
- collect and label more real clips
- quantify override frequency
- run stronger validation on real data

## Short Presentation-Safe Summary
KitchenMaster is a configurable research pipeline for pickleball NVZ foot-fault analysis that combines court registration, ball-based event inference, foot localization, signed-distance classification, and human-in-the-loop review.
