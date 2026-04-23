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

### CourtGeometryModel

`src/court_model.py` — derives all court structure from 6 anchor points:
- `near_kitchen_line`, `far_kitchen_line` — NVZ lines (inferred at 7/22 from net unless explicitly annotated)
- `outer_polygon`, `net_line`, `left_sideline`, `right_sideline`
- `near_legal_polygon`, `far_legal_polygon` — legal zone fills
- `model.warp(H)` — propagates entire model through a homography

### FrameStabilizer

`src/stabilizer.py` — ORB + BFMatcher + RANSAC homography:
- `set_reference(frame)` — detects ORB features in reference frame
- `estimate_transform(frame) → (H, info)` — per-frame homography with Lowe ratio test
- Sanity gate: rejects transforms with >80px translation or det deviation >0.25
- Falls back to previous valid H if estimation fails

---

## Phase 2 (planned) — Foot localization + event detection

Placeholders exist in `src/foot_localizer.py` and `src/event_detector.py`.
These will be connected to the registered court geometry in Phase 2.

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
  foot_localizer.py             Phase 2 placeholder
  event_detector.py             Phase 2 placeholder
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
  run_sim.py                    Phase 0 synthetic pipeline
  run_eval.py                   re-evaluate from saved predictions
  run_real.py                   Phase 0 real data eval
  run_court_registration.py     Phase 1 v1 pipeline (static horizontal line)
  run_court_registration_v2.py  Phase 1 v2 pipeline (ORB homography, no anchor model)
  run_court_registration_v3.py  Phase 1 v3 pipeline (anchor model + ORB) ← current
results/
  sim_v1/                       synthetic pipeline outputs
  real_baseline/court_reg_v1/   Phase 1 v1 outputs
  real_baseline/court_reg_v3/   Phase 1 v3 outputs (after annotation + run)
tests/                          27 unit tests
```

## Reproducibility

All experiment outputs under `results/` are generated by code from config.
Raw video and frames are gitignored. Re-generate outputs by running the pipeline scripts.
