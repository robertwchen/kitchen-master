# Claude Context: KitchenMaster Final Report

You are helping write a short IEEE-style final report for KitchenMaster, a pickleball kitchen/NVZ foot-fault detection research prototype. Use this handoff folder as source material. Prioritize accuracy and do not overclaim.

## Project In One Paragraph

KitchenMaster analyzes pickleball non-volley-zone foot faults from a fixed side-view camera. The pipeline registers court/NVZ geometry, tracks the ball, infers volley/bounce timing, localizes the relevant foot, computes signed distance from the foot to the kitchen boundary, and labels events as `legal`, `foot_fault_volley`, or `uncertain`. The design is deliberately interpretable and human-in-the-loop: ambiguous frames are reviewed rather than forcing confident wrong calls.

## Strong Claims Supported By Repo Outputs

- Synthetic baseline: 200 synthetic samples; legal P/R 1.0/0.94; fault P/R 0.5051/1.0; uncertain P/R 0.9444/0.51; uncertain rate 27.0%; false fault 0.0%; missed fault 0.0%.
- Real court registration v3: court_reg_v3 processed 2055 frames at 59.943 fps; 2055/2055 frames OK; 0 fallbacks; method anchor-point court model + ORB post_translation; left/right boundary edge strength means 12.14/7.31.
- Registration comparison: post-translation had 0 fallback frames; affine had 19 fallback frames.
- Demo pipeline: demo_v1_active_side produced 3 events with labels {'foot_fault_volley': 1, 'uncertain': 2}, threshold 5.0 px and uncertainty margin 15.0 px.

## Caveats To Include

- This is not a production-ready autonomous referee.
- Court registration is the strongest result.
- Ball tracking/event inference and foot localization are still fragile and review-dependent.
- Real foot-fault evaluation size is small; use demo events as qualitative evidence, not broad accuracy proof.

## Suggested Paper Structure

1. Abstract: problem, interpretable pipeline, v3 registration result, synthetic conservative baseline, reviewable demo outputs.
2. Introduction: NVZ foot faults require geometry; uncertainty is preferable to wrong calls.
3. Methods: court model/registration, ball/event inference, foot localization, signed-distance decision, review loop.
4. Experiments: synthetic baseline, registration versions/comparison, demo pipeline events.
5. Results: synthetic metrics, v3 registration 2055/2055, demo event labels/figures.
6. Limitations/Future Work: more labeled real clips, dedicated pickleball detector, improved event timing, quantified override rate.

## Figure Paths

- `figures/photos_for_slides/01_registration_overlay.png`
- `figures/photos_for_slides/02_registration_comparison.png`
- `figures/photos_for_slides/03_detected_fault_event.png`
- `figures/photos_for_slides/04_uncertain_event.png`
- `figures/photos_for_slides/05_uncertain_review_event.png`
- `generated_charts/registration_fallback_comparison.svg`
- `generated_charts/synthetic_key_rates.svg`
- `generated_charts/demo_event_label_counts.svg`

## Key Source Paths

- `source_snapshot/experiments/configs/demo_pipeline.yaml`
- `source_snapshot/experiments/run_demo_pipeline.py`
- `source_snapshot/experiments/run_court_registration_v3.py`
- `source_snapshot/src/court_model.py`
- `source_snapshot/src/stabilizer.py`
- `source_snapshot/src/ball_tracker.py`
- `source_snapshot/src/volley_classifier.py`
- `source_snapshot/src/foot_localizer.py`
- `source_snapshot/src/foot_fault_pipeline.py`
- `source_snapshot/src/evaluate.py`

Write in an IEEE conference style. For a 2-page undergraduate report, stay concise: one compact system overview figure, one small metrics table, and two qualitative result images may be enough.
