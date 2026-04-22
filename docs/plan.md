# Research Plan

## Timeline (3 Days)

### Day 1 — Synthetic data + baseline detector

- Define sim parameters in `experiments/configs/sim_v1.yaml`
- Implement `src/sim_generator.py`: 4 scenario types, seeded, deterministic
- Implement `src/baseline_detector.py`: Hough line detection + foot overlap logic
- Run end-to-end on 200 synthetic frames
- Save confusion matrix and metrics CSV to `results/sim_v1/`

### Day 2 — Real data collection

- Record short clips from a fixed side-view angle (phone or webcam)
- Export frames to `data/real/frames/`
- Label each frame: `legal`, `fault`, or `uncertain`
- Save labels to `data/real/labels.csv`
- Run baseline detector on real frames, save to `results/real_v1/`

### Day 3 — Evaluation and writeup

- Compare sim vs. real performance (metrics side by side)
- Document failure cases: which scenarios broke the detector and why
- Analyze sensitivity to blur, distance, and viewpoint
- Finalize confusion matrices and precision/recall tables
- Write preliminary results section of the report

## Design Decisions

- **Classical CV baseline**: Use edge detection and Hough transforms before considering any learned model. Interpretability matters for v1.
- **Uncertain output**: Margin-based uncertainty instead of forcing every frame into legal/fault. This is the key research hypothesis (RQ3).
- **Config-driven**: All thresholds and paths live in YAML. Changing a threshold should never require editing source code.
- **No fabricated outputs**: Every metric in `results/` comes from actually running the code on real or generated data.

## Known Limitations for v1

- Single static camera — no multi-view triangulation
- Frame-by-frame only — no temporal modeling or event detection
- Foot detection in sim is proxy-based (color + shape); real footage will require adaptation
- Real dataset volume will be small (likely <100 labeled frames)
- Line detection assumes a clear horizontal line; shadows or worn lines will degrade performance

## Next Steps (Post-Prototype)

- Temporal sliding window around volley events
- Real-time inference on video stream
- Improved foot segmentation for real footage
- Larger annotated real dataset
