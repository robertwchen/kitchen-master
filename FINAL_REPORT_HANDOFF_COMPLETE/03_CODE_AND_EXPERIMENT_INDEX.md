# Code And Experiment Index

## Core Pipeline

- `source_snapshot/src/court_model.py`: anchor-based court model. Derives near/far kitchen lines, left/right NVZ boundaries, and legal-side polygons.
- `source_snapshot/src/stabilizer.py`: ORB feature extraction, matching, RANSAC transforms, ROI masking, sanity checks, fallback logic.
- `source_snapshot/src/court_registration.py`: earlier line model and signed-distance primitives.
- `source_snapshot/src/ball_tracker.py`: classical/YOLO-assisted ball tracking, temporal linking, smoothing, CSV and overlay export.
- `source_snapshot/src/ball_detector.py`: Ultralytics wrapper for sports-ball detections.
- `source_snapshot/src/volley_classifier.py`: smooths ball trajectories and exports bounce/event candidates and montages.
- `source_snapshot/src/foot_localizer.py`: background subtraction, ROI thresholding, manual point, and event-hybrid foot localization.
- `source_snapshot/src/foot_fault_pipeline.py`: loads registration, infers active side, localizes foot, computes signed distance, assigns legal/fault/uncertain labels.
- `source_snapshot/src/evaluate.py`: metric and failure-analysis code.
- `source_snapshot/src/sim_generator.py`: synthetic data generation.
- `source_snapshot/src/baseline_detector.py`: interpretable Hough/HSV baseline classifier.

## Experiment Entrypoints

- `source_snapshot/experiments/run_sim.py`: generates synthetic samples, runs the baseline detector, and writes metrics/overlays.
- `source_snapshot/experiments/run_eval.py`: reevaluates saved predictions.
- `source_snapshot/experiments/run_real.py`: applies baseline detector to labeled real frames.
- `source_snapshot/experiments/run_court_registration.py`: v1 static line registration.
- `source_snapshot/experiments/run_court_registration_v2.py`: ORB-based v2 registration.
- `source_snapshot/experiments/run_court_registration_v3.py`: current best anchor model plus ORB/post-translation registration.
- `source_snapshot/experiments/run_demo_pipeline.py`: end-to-end human-in-the-loop demo orchestrator.

## Configs To Cite

- `source_snapshot/experiments/configs/sim_v1.yaml`: 200 synthetic samples, 320x240, detector thresholds.
- `source_snapshot/experiments/configs/court_reg_v1.yaml`: initial real-video registration.
- `source_snapshot/experiments/configs/court_reg_v2.yaml`: ORB/homography registration experiment.
- `source_snapshot/experiments/configs/court_reg_v3.yaml`: current best real registration settings.
- `source_snapshot/experiments/configs/demo_pipeline.yaml`: full pipeline settings for ball tracking, event inference, foot localization, signed-distance labeling, and review workflow.

## Tests

- `source_snapshot/tests/test_sim_generator.py`: synthetic sample generation behavior.
- `source_snapshot/tests/test_detector.py`: detector/classification behavior.
- `source_snapshot/tests/test_evaluate.py`: metrics and evaluation behavior.
