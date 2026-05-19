# Final Report Brief

## Project Thesis

`KitchenMaster` is a research prototype for detecting pickleball kitchen / non-volley-zone (NVZ) foot faults from a single fixed side-view camera. The system is intentionally interpretable: register the court, identify event timing, localize the relevant foot, compute signed distance to the NVZ boundary, and return `legal`, `foot_fault_volley`, or `uncertain`.

## Research Questions

1. Can a fixed side-view camera register the NVZ boundary well enough to support foot-fault decisions?
2. How sensitive is the pipeline to viewpoint, blur, occlusion, ball/event ambiguity, and foot-line distance?
3. Does an explicit `uncertain` output reduce wrong confident calls in ambiguous cases?

## System Overview

The current pipeline is: annotation -> court registration -> ball tracking -> bounce/volley cues -> active-side inference -> foot localization -> signed-distance decision -> human review. The strongest implementation path is configured through `experiments/configs/demo_pipeline.yaml` and orchestrated by `experiments/run_demo_pipeline.py`.

Key modules:

- `src/court_model.py`: anchor-based visible court/NVZ geometry.
- `src/stabilizer.py`: ORB feature matching, RANSAC transform estimation, ROI masks, fallback logic.
- `src/ball_tracker.py` and `src/ball_detector.py`: classical and optional Ultralytics ball tracking.
- `src/volley_classifier.py`: smoothed trajectory and bounce/event candidates.
- `src/foot_localizer.py`: event-hybrid foot localization using pose/lower-body, background subtraction, and ROI cues.
- `src/foot_fault_pipeline.py`: active-side inference, signed distance, thresholds, CSV/review exports.

## Main Results To Use

### Court Registration v3

- Video: `pickle_vid_1_trimmed_from_8s.mp4` at `1920x1080`, `59.943` fps, `2055` frames, `34.28` s.
- Method: `anchor-point court model + ORB post_translation`.
- Success: `2055/2055` frames ok.
- Fallbacks: `0` (`0.0` fallback rate).
- Validation samples: `60` frames.
- Left boundary edge strength mean: `12.14`.
- Right boundary edge strength mean: `7.31`.
- Transform translation mean/max: `25.03` / `202.44` px.
- Comparison: `post_translation` = `0` fallbacks; affine = `19` fallbacks.

### Earlier Registration v1

- Video: `pickle_vid_1.MOV`, `2535` frames, `42.29` s.
- Edge strength mean/std: `51.58` / `29.54`.
- CV: `0.5728`.
- Important narrative: v1/v2 showed that raw strong line detection can lock onto the wrong court structure; v3 fixed this by seeding geometry from verified anchors.

### Synthetic Baseline

- Samples: `200`.
- Legal precision/recall: `1.0` / `0.9`.
- Fault precision/recall: `0.5319` / `1.0`.
- Uncertain precision/recall: `0.918` / `0.56`.
- Uncertain rate: `0.305`.
- False fault rate: `0.0`.
- Missed fault rate: `0.0`.

### Demo Foot-Fault Snapshot

- Active-side run: `3` events, labels `{'foot_fault_volley': 1, 'uncertain': 2}`.
- Foot-review variant: `3` events, labels `{'foot_fault_volley': 2, 'uncertain': 1}`.
- Fault threshold: `5.0` px.
- Uncertain margin: `15.0` px.
- Foot mode: `event_hybrid`.
- Event rows are copied at `data_results/photos_for_slides/demo_foot_fault_events.csv`.

## Recommended Paper Structure

1. Introduction: pickleball NVZ calls are geometry-sensitive; a wrong confident fault call is worse than abstaining.
2. System Overview: present the modular pipeline instead of claiming an end-to-end black-box referee.
3. Methods: court model and registration, ball/event inference, foot localization, signed-distance decision, review loop.
4. Experiments: synthetic baseline, registration v1/v3 comparison, presentation demo events.
5. Results: emphasize v3 registration success, synthetic conservative classification, and demo event examples.
6. Limitations: ball/event timing and foot localization remain review-dependent; real evaluation size is small; some generated videos/CSVs need regeneration.
7. Future Work: dedicated pickleball detector, more labeled real clips, override-rate quantification, stronger event timing, more rigorous real-world validation.

## Figure Suggestions

- Fig. 1: `assets/photos_for_slides/01_registration_overlay.png` - registered court overlay.
- Fig. 2: `assets/photos_for_slides/02_registration_comparison.png` - registration comparison.
- Fig. 3: `assets/photos_for_slides/03_detected_fault_event.png` - detected foot-fault example.
- Fig. 4: `assets/photos_for_slides/04_uncertain_event.png` or `05_uncertain_review_event.png` - uncertainty/human-review example.
- Optional chart: `generated_charts/registration_fallback_comparison.svg`.

## Honest Framing

The report should not claim a production-ready automatic referee. The strongest defensible claim is that the project produced an interpretable, configurable research pipeline and demonstrated robust court registration plus reviewable foot-fault event outputs on a real video clip.
