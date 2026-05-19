# KitchenMaster

Research prototype for pickleball NVZ foot-fault detection from a fixed side-view camera.

## Research Questions

- **RQ1**: Can a fixed side-view camera detect NVZ line contact in controlled conditions?
- **RQ2**: How sensitive is detection to viewpoint, blur, occlusion, and foot-line distance?
- **RQ3**: Can an uncertain output reduce wrong calls in ambiguous cases?

## Output Labels

| Label | Meaning |
|-------|---------|
| `legal` | Foot clearly behind the line |
| `fault` | Foot touches or crosses the line |
| `uncertain` | Cannot be determined reliably |

---

## Phase 1 — Court Registration (real video)

**Goal:** Prove that court geometry (NVZ kitchen lines) can be reliably registered on real footage before any foot-fault classification is attempted.

### Step 1 — Extract reference frames

```bash
python scripts/extract_frames.py \
    --video data/real/videos/pickle_vid_1.MOV \
    --out   data/real/frames/ \
    --fps   5
```

Saves `data/real/frames/<stem>_frameNNNNN.jpg` and `manifest.csv`.

### Step 2 — Annotate kitchen line endpoints

**Option A — Interactive (recommended):**
```bash
python scripts/annotate_reference.py \
    --video data/real/videos/pickle_vid_1.MOV \
    --frame 60 \
    --out   data/real/annotations/annotations.json
```

Click order: near kitchen line p1 → p2, far kitchen line p1 → p2, legal-side reference point.
Keys: `U` undo, `R` reset, `S` save, `Q` quit.

**Option B — Edit JSON manually:**
Copy `data/real/annotations/annotations_template.json`, fill in pixel coordinates
from the reference frame images saved in `data/real/annotations/`.

Current annotations (`annotations.json`) were auto-derived from Hough line detection
on frame 599 (t=10s) and visually verified. Refine with the tool above if needed.

### Step 3 — Run court registration

```bash
python experiments/run_court_registration.py \
    --config experiments/configs/court_reg_v1.yaml
```

**Outputs** (`results/real_baseline/court_reg_v1/`):

| File | Description |
|------|-------------|
| `line_params.csv` | Per-frame line parameters (constant for static camera, 2535 rows) |
| `summary_report.json` | Line equations, refinement offset, stability stats |
| `debug_frames/frame_NNNNN.png` | Annotated overlay images at selected frames |
| `overlay.mp4` | Annotated overlay video (960×540, 10fps) |

### Phase 1 Results (court_reg_v1)

| Property | Value |
|----------|-------|
| Video | pickle_vid_1.MOV — 1920×1080, 59.94fps, 2535 frames (42.3s) |
| Near kitchen line | y = 469px, spans x=[0, 950] |
| Far kitchen line | y = 469px, spans x=[960, 1919] |
| Refinement offset | 0px (annotations aligned well) |
| Edge strength (mean/std) | 51.6 / 29.5 (CV=0.57) |
| Stability note | High CV is expected — players periodically occlude the line; camera geometry is fixed |

---

## Phase 1 v3 — Anchor-Point Court Model + ORB Homography (current)

**Root cause of v1/v2 failure:** The Hough-detected horizontal line at y=469 was the
net top or a tennis court service line, not the pickleball kitchen line. Court geometry
must be seeded from manually-verified anchor points, not raw line detection.

**Camera geometry:** The camera views from one end of the court. In image coordinates:
- Near kitchen line (NVZ) = front **horizontal blue line** — between camera and net
- Far kitchen line (NVZ) = back **horizontal blue line** — behind the net
- Sidelines = slanted lines connecting the corners
- Net = vertical structure in the center

### Step 1 — Annotate anchor points

```bash
python scripts/annotate_anchors.py \
    --video data/real/videos/pickle_vid_1.MOV \
    --frame 0 \
    --out   data/real/annotations/annotations_v3.json
```

Click 6 required anchors in this order (4 more optional, 1 legal-ref):
1. `near_left`  — bottom-left corner of pickleball court (near camera)
2. `near_right` — bottom-right corner (near camera)
3. `net_left`   — left anchor of net
4. `net_right`  — right anchor of net
5. `far_left`   — far-left corner (behind net)
6. `far_right`  — far-right corner (behind net)
7–10. Kitchen-line corners (optional — override 7/22 interpolation)
11. `legal_ref_near` — a point clearly behind the near kitchen line

Keys: `P` preview geometry · `U` undo · `R` reset · `S` save · `Q` quit

### Step 2 — Run v3 registration

```bash
python experiments/run_court_registration_v3.py \
    --config experiments/configs/court_reg_v3.yaml
```

**Outputs** (`results/real_baseline/court_reg_v3/`):

| File | Description |
|------|-------------|
| `per_frame_transforms.csv` | Per-frame H matrix + warped anchor positions + kitchen line endpoints |
| `summary_report.json` | Anchors, registration stats, validation |
| `debug_frames/frame_NNNNN.png` | Annotated overlays at selected frames |
| `overlay.mp4` | Annotated overlay video |
| `feature_roi_mask.png` | Feature-detection ROI used by ORB |
| `comparisons/*/frame_NNNNN.png` | Side-by-side debug comparisons across stability settings |

### Reprojection Validation

Annotate 10-20 sampled frames with the same kitchen-corner anchors:

```bash
python scripts/annotate_reprojection_anchors.py \
    --video data/real/videos/pickle_vid_1_trimmed_from_8s.mp4 \
    --reference-annotations data/real/annotations/annotations_v3.json \
    --n-samples 12 \
    --out data/real/annotations/reprojection_labels_v1.json
```

Then point `validation.reprojection_labels_path` at that JSON and rerun `court_reg_v3`.
The run will write `reprojection_errors.json` and include mean / median / max anchor error in the summary.

### CourtGeometryModel

`src/court_model.py` — derives all court structure from kitchen-corner anchors:
- `near_kitchen_line`, `far_kitchen_line` — front/back edges of the visible kitchen
- `left_boundary_line`, `right_boundary_line` — the key NVZ foot-fault boundaries
- `left_legal_polygon`, `right_legal_polygon` — legal-zone fills
- `model.warp(H)` — propagates the geometry through an affine transform or homography

### FrameStabilizer

`src/stabilizer.py` — ORB + BFMatcher + RANSAC transform estimation:
- `set_reference(frame)` — detects ORB features in reference frame
- `set_feature_mask(mask)` — restricts feature detection/matching to a court ROI
- `estimate_transform(frame) → (H, info)` — affine or homography with Lowe ratio test
- Sanity gate: rejects transforms with >80px translation or det deviation >0.25
- Supports both fixed-reference and rolling-reference tracking
- Falls back to previous valid transform if estimation fails

---

## Phase 2 — Presentation Demo Pipeline (end-to-end, human-in-the-loop)

**Goal:** Presentation-ready feasibility demo connecting court registration → ball tracking
→ bounce/volley inference → foot localization → foot-fault decision, with user validation
checkpoints at every stage.

### What was added

| File | Status before | Status now |
|------|---------------|------------|
| `src/foot_localizer.py` | placeholder (NotImplementedError) | 4 modes: `background_subtraction`, `roi_threshold`, `manual_point`, `event_hybrid` |
| `src/event_detector.py` | placeholder (NotImplementedError) | unchanged — superseded by `volley_classifier.py` |
| `src/ball_tracker.py` | did not exist | HSV yellow ball detection, temporal linking, Gaussian trail smoother, CSV + overlay output |
| `src/volley_classifier.py` | did not exist | Smooth trajectory → bounce candidate detection → volley/post_bounce_hit/uncertain classification + 3-panel montages |
| `src/foot_fault_pipeline.py` | did not exist | Load registration CSV or manual override → localize foot → signed distance to NVZ boundary → legal_volley/foot_fault_volley/uncertain per event |
| `experiments/configs/demo_pipeline.yaml` | did not exist | Full YAML config for all 5 stages + pipeline mode |
| `experiments/run_demo_pipeline.py` | did not exist | End-to-end orchestrator with `auto_review` / `apply_overrides` modes |

### Design: human-in-the-loop validation

The pipeline runs in two modes to prevent silently wrong assumptions:

**Mode 1 — `auto_review`** (run first):
Executes all 5 stages and exports verification artifacts, then writes
`review/review_pending.json` and **stops**.

**Mode 2 — `apply_overrides`** (run after reviewing):
Reads `review_approved.json` (user-edited copy of `review_pending.json`),
applies all corrections, and produces the final annotated outputs and summary video.

### Quickstart

```bash
# Step 1: Run auto_review — generates artifacts and review_pending.json
python experiments/run_demo_pipeline.py \
    --config experiments/configs/demo_pipeline.yaml

# Step 2: Copy and review
cp results/presentation_demo/demo_v1/review/review_pending.json \
   results/presentation_demo/demo_v1/review/review_approved.json
# Edit review_approved.json:
#   - Set each checkpoint's "status" to "approved"
#   - Add user_label / override fields where the system was wrong
#   - See schema in experiments/run_demo_pipeline.py module docstring

# Step 3: Apply overrides and produce final outputs
python experiments/run_demo_pipeline.py \
    --config experiments/configs/demo_pipeline.yaml \
    --mode apply_overrides

# Run only specific stages (e.g. just ball tracking + bounce classification)
python experiments/run_demo_pipeline.py \
    --config experiments/configs/demo_pipeline.yaml \
    --stages 2,3
```

### Validation checkpoints

| Checkpoint | What to inspect | What you can override |
|------------|----------------|-----------------------|
| 1. Registration | NVZ boundary line positions on sampled frames | Manual line coordinates (4 kitchen corners) |
| 2. Ball tracking | Detection samples — is the ball marker on the ball? | Per-frame ball_x / ball_y / confidence |
| 3. Bounce candidates | 3-panel montages (before/at/after), vy_before, vy_after, y_position | `user_label`: bounce / no_bounce / uncertain |
| 4. Foot localization | Foot point (orange dot) on annotated frames | `override_foot_x`, `override_foot_y` |
| 5. Final events | Signed distance, foot point, line, system label | `user_label`: legal_volley / foot_fault_volley / uncertain |

### Output structure

```
results/presentation_demo/<run_name>/
  review/
    review_pending.json           generated by auto_review — copy and edit this
    review_approved.json          user-edited copy — drives apply_overrides
    checkpoint_registration/      NVZ line debug frames
    checkpoint_ball_tracking/     sampled detection PNGs
    checkpoint_bounces/           montages per bounce candidate
    checkpoint_foot/              full-frame + zoomed foot-ROI review frames
    checkpoint_final/             final event annotation frames
  ball_tracking/
    ball_tracking.csv             frame_index, timestamp_s, ball_x, ball_y, confidence
    ball_overlay.mp4              tracking video with trail
    debug_frames/                 sampled annotated PNGs
  volley_events/
    candidates.csv                bounce candidates with vy_before/after, confidence
    events.csv                    classified hit events (if hit_frames supplied)
    montage/                      3-panel PNGs per bounce candidate
  foot_faults/                    (auto_review pass)
    foot_fault_events.csv         per-event signed_dist_px, label, foot_x/y
    event_frames/                 annotated frame per event
    summary.json
  foot_faults_final/              (apply_overrides pass, user corrections applied)
  summary/
    pipeline_summary.json
    court_model_frame0.png
    demo_summary.mp4              summary video (court overlay + ball trail + fault labels)
```

### Manual overrides (demo reliability)

Several override mechanisms are available for presenting reliable demos:

- **NVZ line**: set `registration.manual_override_path` in the YAML, or edit `override` in checkpoint 1 of `review_approved.json`
- **Ball detections**: add entries to `ball_tracking.override_frames` in the review file
- **Bounce labels**: set `user_label` for each candidate in checkpoint 3
- **Foot points**: set `override_foot_x` / `override_foot_y` in checkpoint 4
- **Event labels**: set `user_label` in checkpoint 5
- **Hit frames**: set `foot_fault.manual_volley_frames` in the YAML to hard-code known volley event frames
- **Foot mode**: use `foot_localizer.mode: event_hybrid` for boundary-aware event-frame review, or `manual_point` with `override_file` for fully manual foot annotation

### NVZ boundary geometry (side-facing camera)

From the current annotations (`annotations_v3.json`):
- Near kitchen line: [54, 970] → [1887, 998] (horizontal, near bottom of frame)
- Far kitchen line: [602, 556] → [1290, 570] (horizontal, mid-frame)
- **Left NVZ boundary** (key fault line for left player): [54, 970] → [602, 556]
- **Right NVZ boundary** (key fault line for right player): [1887, 998] → [1290, 570]

`LineModel.signed_distance()` returns positive when the foot is on the legal side
(outside the kitchen) and negative when inside. Set `foot_fault.nvz_side` in the
YAML to `left`, `right`, or `near` depending on which player you are evaluating.

---

## Synthetic Pipeline (Phase 0)

### Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Synthetic pipeline (generate → detect → evaluate)
python experiments/run_sim.py --config experiments/configs/sim_v1.yaml

# Re-evaluate from saved predictions
python experiments/run_eval.py --results results/sim_v1/

# Real labeled frames (after filling annotations.csv)
python experiments/run_real.py \
    --annotations data/real/annotations.csv \
    --results results/real_v1/

# Tests
pytest tests/
```

### Synthetic Results (sim_v1, 200 frames, seed=42)

| Metric | Value |
|--------|-------|
| False fault rate | **0.0%** |
| Missed fault rate | **0.0%** |
| Uncertain rate | **27.0%** |
| Legal P / R | 1.000 / 0.940 |
| Fault P / R | 0.505 / 1.000 |

---

## Implemented Baseline Detector

Classical CV — no learned model:

1. **Line detection**: Canny edges → Hough line transform → median y of horizontal segments
2. **Foot detection**: HSV color mask (green range for sim; adapt hue bounds for real footage)
3. **Classification**: `gap = line_y − foot_bottom`
   - `gap > uncertain_margin_px` → `legal`
   - `gap < −fault_threshold_px` → `fault`
   - otherwise → `uncertain`

---

## Repo Structure

```
docs/                           problem definition and research plan
scripts/
  extract_frames.py             extract frames from video with manifest CSV
  annotate_reference.py         interactive click-to-annotate kitchen line tool (v1)
  annotate_anchors.py           anchor-point annotation tool for court model (v3)
src/
  config.py                     YAML config loader
  court_registration.py         Phase 1 v1 — LineModel + CourtRegistration class
  court_model.py                Phase 1 v3 — CourtGeometryModel from anchor points
  stabilizer.py                 ORB + RANSAC homography frame stabilizer
  viz.py                        overlay drawing: kitchen lines, court model, video writer
  sim_generator.py              synthetic frame generation with SampleMeta
  baseline_detector.py          Hough + HSV + margin classify
  evaluate.py                   metrics, failure analysis, CSV/PNG output
  foot_localizer.py             Phase 2 — bg_subtraction / roi_threshold / manual_point / event_hybrid
  event_detector.py             Phase 2 placeholder (superseded by volley_classifier)
  ball_tracker.py               Phase 2 — HSV yellow ball detection + temporal linking
  volley_classifier.py          Phase 2 — bounce/volley inference + verification montages
  foot_fault_pipeline.py        Phase 2 — NVZ signed distance + fault decision pipeline
data/real/
  videos/                       raw video clips (gitignored)
  frames/                       extracted frames (gitignored)
  annotations/
    annotations.json            current reference annotation
    annotations_template.json   blank template
    reference_frame_*.jpg       reference frames for manual annotation
  annotations.csv               frame-level labels for Phase 0 real eval
experiments/
  configs/
    sim_v1.yaml                 synthetic experiment config
    court_reg_v1.yaml           court registration v1 config (static line)
    court_reg_v2.yaml           court registration v2 config (ORB, no anchor model)
    court_reg_v3.yaml           court registration v3 config (anchor model + ORB)
    demo_pipeline.yaml          Phase 2 end-to-end demo pipeline config
  run_sim.py                    Phase 0 synthetic pipeline
  run_eval.py                   re-evaluate from saved predictions
  run_real.py                   Phase 0 real data eval
  run_court_registration.py     Phase 1 v1 pipeline (static horizontal line)
  run_court_registration_v2.py  Phase 1 v2 pipeline (ORB homography, no anchor model)
  run_court_registration_v3.py  Phase 1 v3 pipeline (anchor model + ORB) ← current
  run_demo_pipeline.py          Phase 2 end-to-end demo pipeline (auto_review / apply_overrides)
results/
  sim_v1/                       synthetic pipeline outputs
  real_baseline/court_reg_v1/   Phase 1 v1 outputs
  real_baseline/court_reg_v3/   Phase 1 v3 outputs (after annotation + run)
tests/                          27 unit tests
```

## Reproducibility

All experiment outputs under `results/` are generated by code from config.
Raw video and frames are gitignored. Re-generate outputs by running the pipeline scripts.
