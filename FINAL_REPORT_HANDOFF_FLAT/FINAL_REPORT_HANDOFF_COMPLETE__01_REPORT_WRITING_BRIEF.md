# Report Writing Brief

## Thesis

KitchenMaster is an interpretable research prototype for pickleball non-volley-zone (NVZ, kitchen) foot-fault analysis from a fixed side-view camera. The system does not try to be a black-box referee. It registers court geometry, tracks the ball, infers event timing, localizes the relevant foot, computes signed distance to the NVZ boundary, and returns `legal`, `foot_fault_volley`, or `uncertain` with review artifacts.

## Recommended 2-Page Story

Use the report to argue feasibility and limitations honestly:

1. The central problem is geometry-sensitive: a foot detector alone is not enough unless the NVZ boundary is correct in the same frame.
2. The strongest technical result is robust real-video court registration using verified court anchors plus ORB/post-translation stabilization.
3. The synthetic experiment demonstrates the conservative decision policy: no false faults and no missed faults in the toy setup, with uncertainty used near ambiguous boundaries.
4. The presentation demo shows the full pipeline is wired end-to-end, but ball/event timing and foot localization remain review-dependent.

## Research Questions

- Can a fixed side-view camera register the kitchen/NVZ boundary well enough to support foot-fault decisions?
- How sensitive is classification to distance from the line, blur, occlusion, viewpoint, and event ambiguity?
- Does an explicit `uncertain` output reduce wrong confident calls near the line?

## Results To Lead With

- Synthetic baseline: 200 synthetic samples; legal P/R 1.0/0.94; fault P/R 0.5051/1.0; uncertain P/R 0.9444/0.51; uncertain rate 27.0%; false fault 0.0%; missed fault 0.0%.
- Court registration v3: court_reg_v3 processed 2055 frames at 59.943 fps; 2055/2055 frames OK; 0 fallbacks; method anchor-point court model + ORB post_translation; left/right boundary edge strength means 12.14/7.31.
- Registration comparison: `post_translation` had 0 fallback frames; affine had 19 fallback frames in the main comparison.
- Demo foot-fault snapshot: demo_v1_active_side produced 3 events with labels {'foot_fault_volley': 1, 'uncertain': 2}, threshold 5.0 px and uncertainty margin 15.0 px.

## Figure Plan

- Fig. 1: `figures/photos_for_slides/01_registration_overlay.png` for court registration overlay.
- Fig. 2: `figures/photos_for_slides/02_registration_comparison.png` or `generated_charts/registration_fallback_comparison.svg` for registration comparison.
- Fig. 3: `figures/photos_for_slides/03_detected_fault_event.png` for a detected event.
- Fig. 4: `figures/photos_for_slides/04_uncertain_event.png` or `05_uncertain_review_event.png` for uncertainty / human review.

## Do Not Overclaim

Do not claim production-ready autonomous refereeing. The defensible claim is a configurable research pipeline with strong court registration, interpretable signed-distance decisions, and reviewable real-video event outputs.
