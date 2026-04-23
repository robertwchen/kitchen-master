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
  annotate_reference.py         interactive click-to-annotate kitchen line tool
src/
  config.py                     YAML config loader
  court_registration.py         Phase 1 — LineModel + CourtRegistration class
  viz.py                        overlay drawing, legal zone shading, video writer
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
    court_reg_v1.yaml           court registration config
  run_sim.py                    Phase 0 synthetic pipeline
  run_eval.py                   re-evaluate from saved predictions
  run_real.py                   Phase 0 real data eval
  run_court_registration.py     Phase 1 court registration pipeline
results/
  sim_v1/                       synthetic pipeline outputs
  real_baseline/court_reg_v1/   Phase 1 outputs (overlay video, line CSV, report)
tests/                          27 unit tests
```

## Reproducibility

All experiment outputs under `results/` are generated by code from config.
Raw video and frames are gitignored. Re-generate outputs by running the pipeline scripts.
