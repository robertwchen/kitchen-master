# Research Plan

## Status

| Stage | Status |
|-------|--------|
| Synthetic data generator with SampleMeta | Done |
| Baseline detector (Hough + HSV + margin classify) | Done |
| Metadata CSV (per-sample ground-truth parameters) | Done |
| Predictions, metrics, confusion matrix CSV + PNG | Done |
| Failure analysis grouped by scenario/occlusion/blur/distance | Done |
| Qualitative overlay images | Done |
| Real-data workflow (annotations.csv → run_real.py) | Scaffolded — awaiting real clips |
| Real data collection and labeling | Day 2 |
| Sim vs. real comparison | Day 3 |

## Timeline

### Day 1 — Synthetic pipeline (complete)

- 4 scenario types: clear_legal, clear_fault, borderline, occluded
- Per-sample SampleMeta: foot position, line position, signed_distance_px, occlusion_flag, blur_level, seed
- Baseline detector: Hough line detection + HSV foot segmentation + margin-based classify
- All outputs saved to `results/sim_v1/`
- 27 unit tests passing

### Day 2 — Real data collection

- Record short clips from a fixed side-view angle (phone or webcam, tripod)
- Export frames: `ffmpeg -i clip.mp4 -vf fps=5 data/real/frames/clip_frame%03d.jpg`
- Label each frame in `data/real/annotations.csv`
- Run `python experiments/run_real.py`
- Document coverage: how many frames, scenario breakdown

### Day 3 — Evaluation + writeup

- Compare sim vs. real metrics side by side
- Failure case analysis: which scenarios and distance ranges break the detector
- Sensitivity study: vary `uncertain_margin_px` and `fault_threshold_px`, observe tradeoffs
- Document what the overlays show — where the detector fails and why
- Write preliminary results section (honest, citing actual numbers)

## Design Decisions

- **Classical CV baseline**: Hough transform + HSV segmentation. No learned model for v1.
  Interpretable and debuggable — every failure can be traced to a specific detector step.
- **Uncertain output**: Margin-based uncertainty rather than forced binary. Directly tests RQ3.
  The `uncertain_margin_px` threshold is the key hyperparameter.
- **SampleMeta with signed distance**: Enables failure analysis by distance bucket, which
  quantifies how sensitivity scales with proximity to the line.
- **Per-sample seed**: Every synthetic sample is reproducible from its stored seed.
- **Config-driven**: Thresholds and paths are in YAML. No hardcoded values in source.

## Key Findings So Far (Sim v1)

From `failure_analysis.csv`:

- **clear_legal, > +20 px**: 100% accuracy — far legal calls are trivial
- **clear_legal, +3 to +10 px**: 0% accuracy — edge legal calls near margin fall into uncertain (expected)
- **borderline, -3 to +3 px**: 79% accuracy — detector handles near-line well with margin
- **borderline, -10 to -3 px**: 0% accuracy — these are labeled uncertain but predicted fault
- **occluded, < -10 px**: 0% accuracy — blur destroys foot detection, detector calls fault by gap

The last two rows are the most important: motion blur turns uncertain (ground truth) into false
faults. This is the core reliability problem RQ2 is testing.

## Known Limitations for v1

- Single static camera — no multi-angle triangulation
- Frame-by-frame only — no temporal event detection
- Foot detection in sim is a green rectangle proxy; real footage will require HSV re-tuning
- Real dataset volume will be small (likely <100 labeled frames)
- Line detection assumes a clear, unoccluded horizontal line

## Next Steps (Post-Prototype)

- Temporal sliding window around the volley event frame
- Foot detection re-tuning for real footage colors
- Larger annotated real dataset
- Sensitivity sweep: `uncertain_margin_px` vs. false_fault_rate tradeoff curve
