# Results Deep Dive

## Synthetic Baseline

Source files: `results/sim_v1/metrics.csv`, `results/sim_v1/confusion_matrix.csv`, `results/sim_v1/failure_analysis.csv`.

Summary: 200 synthetic samples; legal P/R 1.0/0.94; fault P/R 0.5051/1.0; uncertain P/R 0.9444/0.51; uncertain rate 27.0%; false fault 0.0%; missed fault 0.0%.

Confusion matrix rows are true labels and columns are predicted labels:

- legal: 47 legal, 0 fault, 3 uncertain.
- fault: 0 legal, 50 fault, 0 uncertain.
- uncertain: 0 legal, 49 fault, 51 uncertain.

Interpretation: the baseline does not produce false faults on true legal samples, but uncertain true cases often become fault. This supports the report framing that near-line ambiguity is hard and should be treated conservatively.

Failure-analysis slices:

- `borderline`, occlusion=False, blur=0, distance `+3 to +10`: n=13, accuracy=1.0, legal/fault/uncertain preds=0/0/13
- `borderline`, occlusion=False, blur=0, distance `-10 to -3`: n=9, accuracy=0.0, legal/fault/uncertain preds=0/9/0
- `borderline`, occlusion=False, blur=0, distance `-3 to +3`: n=28, accuracy=0.7857, legal/fault/uncertain preds=0/6/22
- `clear_fault`, occlusion=False, blur=0, distance `-10 to -3`: n=13, accuracy=1.0, legal/fault/uncertain preds=0/13/0
- `clear_fault`, occlusion=False, blur=0, distance `< -10`: n=37, accuracy=1.0, legal/fault/uncertain preds=0/37/0
- `clear_legal`, occlusion=False, blur=0, distance `+10 to +20`: n=22, accuracy=1.0, legal/fault/uncertain preds=22/0/0
- `clear_legal`, occlusion=False, blur=0, distance `+3 to +10`: n=3, accuracy=0.0, legal/fault/uncertain preds=0/0/3
- `clear_legal`, occlusion=False, blur=0, distance `> +20`: n=25, accuracy=1.0, legal/fault/uncertain preds=25/0/0
- `occluded`, occlusion=True, blur=1, distance `+3 to +10`: n=10, accuracy=1.0, legal/fault/uncertain preds=0/0/10
- `occluded`, occlusion=True, blur=1, distance `-10 to -3`: n=17, accuracy=0.0, legal/fault/uncertain preds=0/17/0
- `occluded`, occlusion=True, blur=1, distance `-3 to +3`: n=10, accuracy=0.6, legal/fault/uncertain preds=0/4/6
- `occluded`, occlusion=True, blur=1, distance `< -10`: n=13, accuracy=0.0, legal/fault/uncertain preds=0/13/0

## Real Court Registration

v1 source: `results/real_baseline/court_reg_v1/summary_report.json`.

- Video: `pickle_vid_1.MOV`, frames `2535`, duration `42.29` s.
- v1 line estimate: near/far horizontal line around y=469 px.
- Edge strength mean/std/CV: `51.58` / `29.54` / `0.5728`.
- Narrative: v1 was useful but could lock onto the wrong visual structure, motivating anchor-based geometry.

v3 source: `results/real_baseline/court_reg_v3/summary_report.json`.

- court_reg_v3 processed 2055 frames at 59.943 fps; 2055/2055 frames OK; 0 fallbacks; method anchor-point court model + ORB post_translation; left/right boundary edge strength means 12.14/7.31.
- Reference anchors: `{'kitchen_near_left': [54.0, 970.0], 'kitchen_near_right': [1887.0, 998.0], 'kitchen_far_left': [602.0, 556.0], 'kitchen_far_right': [1290.0, 570.0]}`.
- Stabilizer settings: `{'n_features': 4000, 'ratio_test': 0.75, 'min_matches': 15, 'ransac_threshold_px': 4.0, 'transform_type': 'post_translation', 'rolling_reference': False, 'max_translation_px': 80.0, 'max_rotation_deg': None, 'max_scale_dev': None, 'top_mask_frac': 0.18, 'bottom_mask_frac': 0.1, 'roi': {'enabled': True, 'fill_court': False, 'expand_scale_x': 1.08, 'expand_scale_y': 1.18, 'line_band_px': 44, 'court_padding_px': 12}, 'translation_tracker': {'annotations_path': 'data/real/annotations/annotations_v2.json', 'anchor_name': 'net_base_center', 'template_half_size_px': 24, 'search_radius_px': 36, 'use_previous_match': True, 'min_score': 0.55}}`.
- Validation assessment: `check`.

Registration comparison source: `results/real_baseline/court_reg_v3/comparisons/comparison_report.json`.

- `post_translation_vs_affine_fixed`: post-translation 2055 OK / 0 fallback; affine 2036 OK / 19 fallback.
- `post_translation_vs_static`: both report 2055 OK / 0 fallback, but post-translation models frame drift while static assumes no movement.
- `refinement_on_vs_off`: affine with refinement on/off both report 2036 OK / 19 fallback in this snapshot.

## Demo Pipeline

Sources: `results/presentation_demo/*/review/review_pending.json`, `results/presentation_demo/*/foot_faults/summary.json`, and `figures/photos_for_slides/demo_foot_fault_events.csv`.

- Base `demo_v1` review ball detection rate: `94.5` percent; no final events in that pending review.
- Active-side run ball detection rate: `35.7` percent.
- Active-side events: demo_v1_active_side produced 3 events with labels {'foot_fault_volley': 1, 'uncertain': 2}, threshold 5.0 px and uncertainty margin 15.0 px.
- Foot-review variant: `3` events with labels `{'foot_fault_volley': 2, 'uncertain': 1}`.

Interpretation: the complete pipeline is wired, but the report should emphasize reviewable outputs and human-in-the-loop validation rather than autonomous accuracy.
