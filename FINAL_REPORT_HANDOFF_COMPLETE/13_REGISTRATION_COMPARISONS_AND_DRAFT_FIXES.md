# Registration Comparisons And Draft Fixes

This note is for revising the IEEE report draft. It separates what is supported by saved outputs from claims that should be softened or corrected.

## Biggest Draft Corrections

### Do not say v1 had about 230 fallbacks

The saved `court_reg_v1` report does not define fallbacks. It is a static line-registration baseline with stability/edge-strength metrics:

- Source: `results/real_baseline/court_reg_v1/summary_report.json`
- Video: `pickle_vid_1.MOV`
- Frames: `2535`
- Duration: `42.29 s`
- FPS: `59.943`
- Estimated line: horizontal near/far line at about `y=469 px`
- Edge strength mean/std/CV: `51.58 / 29.54 / 0.5728`
- Overall assessment: `check`

Paper-safe wording:

> The v1 baseline used a static Hough-derived horizontal line and produced a high edge-strength coefficient of variation (CV = 0.573) across sampled frames, indicating unstable visual support under occlusion and lighting variation. It did not provide a robust per-frame geometric model and motivated the later anchor-based approach.

### Do not say frame 929 signed distance was -440 px

The saved active-side event CSV reports:

- Source: `FINAL_REPORT_HANDOFF_COMPLETE/results/presentation_demo/demo_v1_active_side/foot_faults/foot_fault_events.csv`
- Frame `929`
- Timestamp `15.4981 s`
- Label `foot_fault_volley`
- Active side `right`
- Signed distance `-10.78 px`
- Foot confidence `0.705`
- Foot mode `event_hybrid`

Paper-safe wording:

> The clearest active-side demo event occurred at frame 929 (`t = 15.50 s`), where the system selected the right-side boundary and labeled the event `foot_fault_volley` with signed distance `-10.78 px` and foot confidence `0.705`.

Do not use the old `03_detected_fault_event.png` unless you manually regenerate or verify the text overlay. The image text does not match the strongest saved CSV values you should cite.

### Be careful with "homography translation stayed within 18 px"

The saved v3 summary reports transform translation statistics on sampled frames:

- Mean `25.03 px`
- Median `21.37 px`
- Std `25.88 px`
- Min `0.71 px`
- Max `202.44 px`
- n `60`

So do not claim max drift stayed within `18 px`. Safer wording:

> The sampled transform translation had median `21.37 px` and mean `25.03 px`, with occasional larger excursions up to `202.44 px`. Despite these shifts, the post-translation registration reported zero fallback frames.

## What You Did Compare Before The Final Anchor Method

You should mention this more explicitly in the paper. The project did not jump directly to the final court-registration method.

### v1: static Hough-style line baseline

Files:

- `experiments/run_court_registration.py`
- `experiments/configs/court_reg_v1.yaml`
- `results/real_baseline/court_reg_v1/summary_report.json`
- `FINAL_REPORT_HANDOFF_COMPLETE/results/real_baseline/court_reg_v1/debug_frames/*.png`
- `FINAL_REPORT_HANDOFF_COMPLETE/results/real_baseline/court_reg_v1/overlay.mp4`

What it did:

- Used manually/automatically derived near/far line annotations.
- Treated the court line as static across frames.
- Wrote one line row per frame, debug frames, and overlay video.
- Measured edge strength at sampled frames.

Saved result:

- `2535` frames, `42.29 s`, `59.943 fps`.
- Near/far line around `y=469 px`.
- Edge strength CV `0.5728`.
- Overall assessment `check`.

Interpretation:

- It was simple and useful as a baseline, but it could lock onto the wrong visual structure and did not provide robust anchor-based court geometry.
- It motivated moving away from raw line detection toward manually verified court anchors.

### v2: ORB/RANSAC homography registration

Files:

- `experiments/run_court_registration_v2.py`
- `experiments/configs/court_reg_v2.yaml`
- `src/stabilizer.py`

What it tried:

- ORB features, BFMatcher, Lowe ratio test, RANSAC transform estimation.
- `transform_type: homography` in the config.
- Optional line refinement with local Sobel search.
- Comparison against v1 edge strength was supported by the v2 runner design.

Important caveat:

- No current saved `results/real_baseline/court_reg_v2/summary_report.json` exists in the live `results/` snapshot. Discuss v2 qualitatively as an engineering iteration unless you regenerate outputs.

Paper-safe wording:

> A second iteration introduced ORB/RANSAC homography tracking to warp the reference geometry across frames. This improved the idea of frame-to-frame propagation but still lacked a reliable court-specific geometry prior; when visual features were sparse or distracting, the transform alone was not enough to guarantee that the projected NVZ line corresponded to the actual pickleball kitchen boundary.

### v3: anchor-point court model plus stabilization

Files:

- `experiments/run_court_registration_v3.py`
- `experiments/configs/court_reg_v3.yaml`
- `src/court_model.py`
- `src/stabilizer.py`
- `results/real_baseline/court_reg_v3/summary_report.json`
- `results/real_baseline/court_reg_v3/comparisons/comparison_report.json`

What changed:

- Manually verified court anchors define the actual kitchen/NVZ geometry.
- `CourtGeometryModel` derives near/far kitchen lines, left/right NVZ boundaries, and legal-side polygons.
- ORB/RANSAC/translation tracking updates the geometry per frame.
- Feature extraction is restricted to a court/line ROI.
- Net-base template tracking supports the `post_translation` mode.

Saved result:

- `2055/2055` frames OK.
- `0` fallbacks.
- Fallback rate `0.0`.
- Method: `anchor-point court model + ORB post_translation`.
- Left/right boundary edge strength means: `12.14 / 7.31`.
- Overall assessment: `check`.
- Reprojection validation: `null`, because labeled reprojection anchors were not populated.

Paper-safe wording:

> The final registration version used manually verified court anchors to instantiate an explicit `CourtGeometryModel`. Rather than detecting arbitrary strong lines in each frame, the model first defines the expected NVZ geometry and then updates it with a constrained stabilizer. This made the registration interpretable and gave the system a stable geometric prior.

## v3 Internal Method Comparisons

Source:

- `results/real_baseline/court_reg_v3/comparisons/comparison_report.json`
- `FINAL_REPORT_HANDOFF_COMPLETE/results/real_baseline/court_reg_v3/comparisons/*/*.png`
- `FINAL_REPORT_HANDOFF_COMPLETE/figures/photos_for_slides/02_registration_comparison.png`

Saved comparisons:

| Comparison | Left method | Left result | Right method | Right result | Interpretation |
|---|---|---:|---|---:|---|
| `post_translation_vs_affine_fixed` | post-translation, fixed ref, refine off | `2055 OK / 0 fallback` | affine, fixed ref, refine off | `2036 OK / 19 fallback` | Main quantitative evidence for post-translation robustness |
| `post_translation_vs_static` | post-translation | `2055 OK / 0 fallback` | static | `2055 OK / 0 fallback` | Static does not fail by fallback count, but cannot model camera/frame drift |
| `refinement_on_vs_off` | affine refine on | `2036 OK / 19 fallback` | affine refine off | `2036 OK / 19 fallback` | Refinement did not fix the affine fallback issue in this snapshot |

There are also side-by-side PNG folders in the handoff for:

- `homography_vs_affine`
- `rolling_vs_fixed`
- `static_vs_affine_fixed`
- `post_translation_vs_affine_fixed`
- `post_translation_vs_static`
- `refinement_on_vs_off`

However, the current saved `comparison_report.json` only contains quantitative entries for the three comparisons above. If you discuss `homography_vs_affine`, `rolling_vs_fixed`, or `static_vs_affine_fixed`, frame them as visual/debug comparison exports unless you regenerate a JSON summary for them.

Best visual comparison to use:

- `results/real_baseline/court_reg_v3/comparisons/rolling_vs_fixed/frame_02000.png`
- Copied as `FINAL_REPORT_FIGURES_COURT_FAULT_FAILURE/Optional_Figure_3_Registration_Rolling_vs_Fixed_Frame_02000.png`
- This frame is visually stronger than the generic slide comparison because the left panel shows the weaker rolling-reference registration while the right panel shows the more stable fixed-reference result.

## Better Figure Plan

Use these actual pipeline outputs:

1. Figure 1: `FINAL_REPORT_FIGURES_COURT_FAULT_FAILURE/Figure_1_Court_Registration_Overlay.png`
   - Source: `FINAL_REPORT_HANDOFF_COMPLETE/figures/photos_for_slides/01_registration_overlay.png`
   - Shows registered court/NVZ geometry.

2. Figure 2: `FINAL_REPORT_FIGURES_COURT_FAULT_FAILURE/Figure_2_Uncertain_Event_Frame_1948.png`
   - Source: `FINAL_REPORT_HANDOFF_COMPLETE/figures/photos_for_slides/04_uncertain_event.png`
   - Shows an honest uncertain real event and supports the paper's review-prototype story.

3. Optional Figure 3: `FINAL_REPORT_FIGURES_COURT_FAULT_FAILURE/Optional_Figure_3_Registration_Rolling_vs_Fixed_Frame_02000.png`
   - Source: `results/real_baseline/court_reg_v3/comparisons/rolling_vs_fixed/frame_02000.png`
   - Use this if you want to show that multiple registration/stabilization settings were compared. In this frame, the first method is visibly poor while the fixed-reference result is much more stable.

Alternative Optional Figure 3:

- `FINAL_REPORT_FIGURES_COURT_FAULT_FAILURE/Optional_Figure_3_Failure_Montage_Ball_Candidates_Glare.png`
- Use this if you want to emphasize ball-tracking clutter/failure rather than registration comparisons.

## Replacement Results Text For Court Registration

Use this instead of the current paragraph:

> Court registration was evaluated through a sequence of increasingly constrained methods. The first baseline used a static Hough-style line estimate and produced a high edge-strength coefficient of variation (`CV = 0.5728`) on sampled frames, suggesting unstable visual support and possible confusion with non-NVZ line structures. A second iteration introduced ORB/RANSAC homography tracking, but transform estimation alone did not encode which image lines were actually the pickleball kitchen boundaries. The final version used manually verified court anchors to construct an explicit `CourtGeometryModel`, then propagated that model using the stabilizer. In the saved v3 run, the anchor-based post-translation method registered all `2055/2055` frames with `0` fallbacks. Within v3, post-translation had `0` fallbacks compared with `19` fallbacks for the affine comparison (`2036/2055` OK). Static registration also reported `0` fallbacks, but it does not model frame-to-frame drift; therefore, fallback count alone does not prove static alignment quality. This progression supports the conclusion that court geometry should be seeded from trusted anchors rather than inferred from arbitrary strong image lines.

## Replacement Real-Video Demo Text

Use this instead of claiming a `-440 px` signed distance:

> In the active-side demo run, the pipeline evaluated three manually specified volley candidate frames. Frame `929` (`t = 15.50 s`) was labeled `foot_fault_volley` with signed distance `-10.78 px` and foot confidence `0.705`. Frames `1537` and `1948` were labeled `uncertain`; both required review because foot confidence, active-side confidence, or boundary selection was not strong enough for an autonomous call. A separate foot-review variant produced `2` `foot_fault_volley` labels and `1` `uncertain`, illustrating that side/foot review can change event labels. These results should be presented as qualitative end-to-end evidence, not as real-video accuracy.

## Replacement Limitations Text

Use this:

> The main limitation is not the signed-distance classifier itself, but the upstream perception chain. Ball tracking produced no automatic bounce candidates in the full-pipeline benchmark, so the demo foot-fault stage used manually specified event frames. Active-side inference also changed labels across variants. The foot localizer uses YOLOv8n-pose ONNX to find the person and leg, then refines the contact point with local masks and edges, but per-event confidence remains low enough that review is necessary. The system is therefore best framed as an offline review prototype: it can register the court and produce interpretable review artifacts, but it cannot yet autonomously detect every volley and make reliable live calls.

