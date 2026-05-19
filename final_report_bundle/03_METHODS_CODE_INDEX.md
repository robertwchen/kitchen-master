# Methods Code Index

## Court Geometry And Registration

- `source_snapshot/src/court_registration.py`: v1 line model and signed distance primitives.
- `source_snapshot/src/court_model.py`: current anchor-based court geometry model; derives near/far kitchen lines and left/right NVZ boundaries.
- `source_snapshot/src/stabilizer.py`: ORB features, matcher, RANSAC transforms, ROI masks, sanity checks, fallback behavior.
- `source_snapshot/experiments/run_court_registration_v3.py`: current registration experiment runner.
- `source_snapshot/experiments/configs/court_reg_v3.yaml`: current best registration settings.

## Ball And Event Reasoning

- `source_snapshot/src/ball_tracker.py`: diff/HSV tracking, optional Ultralytics backend, temporal linking, smoothing, CSV/overlay export.
- `source_snapshot/src/ball_detector.py`: Ultralytics wrapper for sports-ball detections.
- `source_snapshot/src/volley_classifier.py`: smoothed ball trajectory, bounce candidates, event classification, montage export.

## Foot Localization And Decision

- `source_snapshot/src/foot_localizer.py`: `background_subtraction`, `roi_threshold`, `manual_point`, and `event_hybrid` foot localization.
- `source_snapshot/src/foot_fault_pipeline.py`: loads registration, infers active side, localizes foot, computes signed distance, labels final events.
- `source_snapshot/experiments/run_demo_pipeline.py`: human-in-the-loop orchestrator for auto-review and apply-overrides modes.
- `source_snapshot/experiments/configs/demo_pipeline.yaml`: full end-to-end config.

## Synthetic Baseline And Evaluation

- `source_snapshot/src/sim_generator.py`: synthetic sample generation and metadata.
- `source_snapshot/src/baseline_detector.py`: Hough/HSV classical baseline.
- `source_snapshot/src/evaluate.py`: metrics, failure analysis, CSV/plot helpers.
- `source_snapshot/experiments/run_sim.py`: synthetic run entrypoint.
- `source_snapshot/tests/*.py`: unit tests for detector, generator, and evaluation behavior.

## Placeholder / Experimental Notes

- `source_snapshot/src/event_detector.py` remains a placeholder and is superseded by `volley_classifier.py` in the current demo pipeline.
- Learned ball detection uses generic YOLO sports-ball settings unless a dedicated pickleball model is trained.
