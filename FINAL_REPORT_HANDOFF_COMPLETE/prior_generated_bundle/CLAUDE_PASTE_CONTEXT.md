# Claude Paste Context: KitchenMaster Final Report

You are helping write a 2-page IEEE conference-style final report for a project called KitchenMaster.

## One-Sentence Summary

KitchenMaster is an interpretable computer-vision research prototype for pickleball kitchen/NVZ foot-fault analysis from a fixed side-view camera.

## Core Pipeline

Annotation -> anchor-based court model -> ORB/RANSAC stabilization -> ball tracking -> bounce/volley cueing -> active-side inference -> event-hybrid foot localization -> signed-distance-to-NVZ decision -> human review/overrides.

## Research Questions

1. Can a fixed side-view camera detect NVZ line contact in controlled conditions?
2. How sensitive is detection to viewpoint, blur, occlusion, and foot-line distance?
3. Can an `uncertain` output reduce wrong calls in ambiguous cases?

## Main Quantitative Results

Court registration v3:
- `2055/2055` successful frames on `pickle_vid_1_trimmed_from_8s.mp4`.
- `0` fallbacks, fallback rate `0.0`.
- Post-translation comparison: `0` fallback frames; affine comparison: `19` fallback frames.
- Validation sampled `60` frames; left/right edge means `12.14` / `7.31`.

Synthetic baseline:
- `n=200`.
- Legal precision/recall `1.0` / `0.9`.
- Fault precision/recall `0.5319` / `1.0`.
- Uncertain precision/recall `0.918` / `0.56`.
- False fault rate `0.0`; missed fault rate `0.0`; uncertain rate `0.305`.

Demo foot-fault events:
- Active-side summary: `3` events, labels `{'foot_fault_volley': 1, 'uncertain': 2}`.
- Thresholds: `5.0` px fault threshold, `15.0` px uncertainty margin.

## Strongest Claim

The project demonstrates a working, interpretable, reviewable pipeline and a robust real-video court-registration stage. It should be framed as a research prototype, not a production-ready autonomous referee.

## Best Figures In This Bundle

- `assets/photos_for_slides/01_registration_overlay.png`
- `assets/photos_for_slides/02_registration_comparison.png`
- `assets/photos_for_slides/03_detected_fault_event.png`
- `assets/photos_for_slides/04_uncertain_event.png`
- `assets/photos_for_slides/05_uncertain_review_event.png`
- `generated_charts/registration_fallback_comparison.svg`

## Limitations

Ball tracking and event timing remain fragile; foot localization needs human review in difficult frames; the real-event evaluation scope is small; some generated outputs referenced by JSON are missing and should be regenerated before citing them as included artifacts.
