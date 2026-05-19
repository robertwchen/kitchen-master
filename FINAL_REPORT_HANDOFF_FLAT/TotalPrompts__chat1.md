Create a concise but comprehensive technical architecture inventory for the repo at /Users/robert/dev/git/school/kitchen-master/kitchen-master. Focus on producing a pasteable summary for an external LLM/chat. Cover: 1) top-level directory structure and purpose, 2) all major Python modules and what each does, grouped by subsystem, 3) all experiment runner scripts and what each orchestrates, 4) YAML configs and what knobs they expose, 5) data flow through the current system, 6) outputs/results structure, 7) dependencies/stack, 8) what is implemented vs placeholder. Return as clean markdown with short sections and bullets. Readonly only.

Exploring the repository structure, Python modules, experiment scripts, configs, and documentation.

Here is a paste-ready technical architecture inventory for the repository (read-only pass over /Users/robert/dev/git/school/kitchen-master/kitchen-master).

Kitchen Master — technical architecture inventory
Purpose: Research prototype for pickleball non-volley zone (NVZ) foot-fault reasoning from a fixed side-view camera: court geometry registration, ball tracking, bounce/volley heuristics, foot localization, and fault labeling with a human-in-the-loop review path.

1) Top-level directory structure and purpose
Path	Role
/Users/robert/dev/git/school/kitchen-master/kitchen-master/src/
Core library: geometry, CV pipelines, eval, demo stages
/Users/robert/dev/git/school/kitchen-master/kitchen-master/experiments/
Runnable entrypoints (run_*.py) + experiments/configs/*.yaml
/Users/robert/dev/git/school/kitchen-master/kitchen-master/scripts/
Offline tooling: frame extraction, interactive annotation (v1 lines, v3 anchors, reprojection labels)
/Users/robert/dev/git/school/kitchen-master/kitchen-master/data/
Real/synthetic inputs: data/real/videos/, frames/, annotations/ (many large assets gitignored per .gitignore)
/Users/robert/dev/git/school/kitchen-master/kitchen-master/tests/
pytest unit tests
/Users/robert/dev/git/school/kitchen-master/kitchen-master/docs/
Problem/plan text (e.g. docs/plan.md)
/Users/robert/dev/git/school/kitchen-master/kitchen-master/results/
Default output root for experiments (gitignore patterns for *.png/*.mp4/*.csv under results/)
/Users/robert/dev/git/school/kitchen-master/kitchen-master/misc/
Ad-hoc configs, exports, analysis artifacts (not the canonical config tree)
/Users/robert/dev/git/school/kitchen-master/kitchen-master/models/, yolov8*.pt, Ultralytics/
Weights and Ultralytics app cache (often untracked or local)
README.md, requirements.txt
High-level map and dependencies
2) Major Python modules (by subsystem)
Configuration
/Users/robert/dev/git/school/kitchen-master/kitchen-master/src/config.py — load_config() (YAML) and get_default_config() for synthetic runs.
Court geometry and registration
src/court_registration.py — LineModel (two-point line, signed distance), CourtRegistration (v1: JSON endpoints, stability via Sobel along lines).
src/court_model.py — CourtGeometryModel: anchor-based kitchen rectangle, left/right NVZ boundaries, legal polygons, warp under homography.
src/stabilizer.py — FrameStabilizer: ORB + BF + RANSAC, affine or homography, masks/sanity gates, optional line ROI refinement helpers.
Baseline (Phase 0) synthetic + classical detector
src/sim_generator.py — synthetic frames + SampleMeta (labels, line position, foot box, scenarios).
src/baseline_detector.py — Hough/gradient line y, HSV foot blob, legal / fault / uncertain from vertical gap to line.
src/evaluate.py — metrics (precision/recall per label, false-fault, missed-fault, uncertain rate), confusion matrix, failure tables, overlay plots.
Visualization
src/viz.py — draw kitchen lines / court model, debug frame export, overlay video writing for registration.
Phase 2 demo stack (real video)
src/ball_tracker.py — per-frame ball pipeline: diff + HSV (tunable), temporal linking, smoothing, CSV + overlay + debug frames; composes with ball_detector for Ultralytics path.
src/ball_detector.py — UltralyticsBallDetector (optional YOLO class filter / tiled inference / merge) — lazy ultralytics import.
src/volley_classifier.py — trajectory smoothing, bounce candidates from vertical velocity, optional volley vs post_bounce when hit_frames provided; montages for review.
src/foot_localizer.py — modes: background_subtraction, roi_threshold, manual_point, event_hybrid (MOG2 + threshold + boundary-aware ROI + optional ONNX pose via OpenCV DNN).
src/foot_fault_pipeline.py — load registration CSV (or manual override) → CourtGeometryModel / LineModel → foot point → signed distance to chosen NVZ side → legal_volley / foot_fault_volley / uncertain + review-oriented artifacts.
src/event_detector.py — placeholder: raises NotImplementedError (“scheduled for Phase 2”); superseded in practice by volley_classifier.py.
Package shell
src/__init__.py — package marker (minimal).
Scripts (not src/)
scripts/extract_frames.py — video → frames + manifest.
scripts/annotate_reference.py — v1 line annotation UI.
scripts/annotate_anchors.py — v3 anchor annotation UI.
scripts/annotate_reprojection_anchors.py — multi-frame anchor labels for reprojection error validation.
Tests
tests/test_*.py — detector, eval, sim generator (and related).
3) Experiment runner scripts and what they orchestrate
Script	Path	Orchestrates
run_sim.py
experiments/run_sim.py
Synthetic: generate_dataset → baseline_detector.predict → evaluate saves → results/<run_name>/ (metadata, predictions, metrics, plots).
run_eval.py
experiments/run_eval.py
Recomputes metrics from an existing predictions.csv in a results folder.
run_real.py
experiments/run_real.py
Labeled real frames (annotations.csv with frame_path,true_label) → baseline_detector → same eval outputs.
run_court_registration.py
experiments/run_court_registration.py
Phase 1 v1: static CourtRegistration from JSON, per-frame line CSV, debug PNGs, overlay video, summary JSON.
run_court_registration_v2.py
experiments/run_court_registration_v2.py
v2: ORB homography, warped LineModel, optional refinement, optional comparison to v1 summary.
run_court_registration_v3.py
experiments/run_court_registration_v3.py
v3 (current geometry path): CourtGeometryModel + FrameStabilizer, per-frame transforms CSV, validation, optional reprojection, comparison exports, overlay.
run_demo_pipeline.py
experiments/run_demo_pipeline.py
End-to-end demo: registration CSV + video → (optional) ball → volley → foot → foot-fault; auto_review (writes review_pending.json and stops) vs apply_overrides; selective --stages.
4) YAML configs and main knobs
Canonical configs live under /Users/robert/dev/git/school/kitchen-master/kitchen-master/experiments/configs/.

sim_v1.yaml — run_name, sim.* (count, size, line position, foot size, blur), detector.* (thresholds, line_detection: hough), output.* (results dir, plots/overlays/frames).
court_reg_v1.yaml — video, annotations, registration.refine*, output.* (overlay fps/scale, debug frame indices).
court_reg_v2.yaml — stabilizer (ORB/RANSAC, transform type, masks), refinement, comparison (vs v1), output.
court_reg_v3.yaml — stabilizer (features, ratio, RANSAC, transform_type, rolling ref, translation tracker sub-block, ROI bands), refinement, output, validation (sample size, reprojection_labels_path), comparison_exports.
demo_pipeline.yaml — Full pipeline: video (optional clip range), registration (CSV, annotations, optional manual line JSON), ball_tracking (tracking_backend: blob vs ultralytics, detection_mode, HSV/diff/morph/temporal, Ultralytics tile/conf/class_ids), volley_classification (smoothing, bounce gates, hit_frames), foot_localizer (mode + long list: pose ONNX path, BGSub, ROI, thresholds), foot_fault (nvz_side, thresholds, active-side heuristics, manual_volley_frames), pipeline (mode, review JSON paths), output (summary video flags).
Misc duplicates/variants (e.g. /Users/robert/dev/git/school/kitchen-master/kitchen-master/misc/demo_pipeline_active_side.yaml, demo_pipeline_foot_review.yaml) — alternate demo presets; same general schema as demo_pipeline.yaml with run-specific run_name and tuned sections.

5) Data flow (current system)
Annotation (offline): JSON anchors (annotations_v3.json style) and/or reprojection label JSON; optional per-frame foot overrides.
Court registration (v3): Video + reference geometry → per-frame homography/affine → per_frame_transforms.csv + reports/overlays.
Demo pipeline: Reads precomputed registration CSV (does not re-run v3 inside the demo) + same video stream.
Ball: Frame stream → (optional) YOLO candidates + classical gates → linked trajectory → ball_tracking.csv (+ overlay).
Volley/bounce: CSV trajectory → bounce candidates and (if hit_frames) event classification → volley_events/*.
Foot + fault: For each event frame, registration row → CourtGeometryModel / boundary LineModel → foot localization → signed distance → label; review JSON records checkpoints and overrides.
Human loop: auto_review exports artifacts and review_pending.json; user copies/edits to review_approved.json; apply_overrides produces final foot_faults_final/, summary/, and summary video.
6) Outputs / results structure (conventions)
Synthetic / baseline eval: results/<run_name>/ — metadata.csv, predictions.csv, metrics.csv, confusion_matrix.*, failure_analysis.csv, optional qualitative overlays.
Court registration v1/v2/v3: Under results/real_baseline/ (per README): line or transform CSVs, summary_report.json, debug_frames/, overlay.mp4, v3 may include comparisons/, reprojection_errors.json when labels exist.
Demo pipeline: results/presentation_demo/<run_name>/ with review/ (checkpoints + JSON), ball_tracking/, volley_events/, foot_faults/, foot_faults_final/, summary/ (e.g. pipeline_summary.json, demo_summary.mp4) — as documented in README.md and run_demo_pipeline.py docstring.
(Repo .gitignore is written to exclude most generated results/** media/tabular files from version control.)

7) Dependencies / stack
From /Users/robert/dev/git/school/kitchen-master/kitchen-master/requirements.txt:

Core: numpy, pandas, matplotlib, opencv-python, pyyaml, scikit-learn, pytest, ultralytics.
Implicit / runtime: OpenCV DNN for ONNX pose in foot_localizer (no separate onnxruntime in requirements.txt — uses cv2.dnn.readNetFromONNX). Learned path uses Ultralytics YOLO; demo config may reference COCO class sports ball as a stand-in when a custom pickleball checkpoint is absent.

8) Implemented vs placeholder
Item	Status
Synthetic generation + classical baseline + evaluation
Implemented
Court registration v1, v2, v3
Implemented (v3 is the anchor + ORB path described in README)
Stabilizer, court geometry model, viz
Implemented
Ball tracking (classical + optional Ultralytics) and ball_detector
Implemented
Volley / bounce classification
Implemented (heuristic; hit classification depends on hit_frames or manual review)
Foot localization (four modes, hybrid + pose)
Implemented
Foot-fault pipeline + demo orchestrator + review JSON workflow
Implemented
src/event_detector.py
Placeholder (NotImplementedError) — not used as the main event path
“Production” foot/ball: dedicated trained pickleball model
Not assumed; repo uses generic YOLO + COCO class or classical CV; README marks custom checkpoint as future
Paths above are under /Users/robert/dev/git/school/kitchen-master/kitchen-master unless noted as VCS-ignored or local-only.

