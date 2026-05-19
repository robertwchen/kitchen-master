# Reproducibility And Gaps

## Commands From Repo Docs

```bash
python experiments/run_sim.py --config experiments/configs/sim_v1.yaml
python experiments/run_eval.py --results results/sim_v1/
python experiments/run_court_registration_v3.py --config experiments/configs/court_reg_v3.yaml
python experiments/run_demo_pipeline.py --config experiments/configs/demo_pipeline.yaml
python experiments/run_demo_pipeline.py --config experiments/configs/demo_pipeline.yaml --mode apply_overrides
```

## Test Status

Attempted in the active environment:

```bash
pytest tests/
python3 -m pytest tests/
```

Both failed because `pytest` is not installed in the active Python environment. The test files themselves are copied under `source_snapshot/tests/`.

## Missing Or Regenerate

See `missing_or_regenerate.csv`. The most important absent files are:

- `results/real_baseline/court_reg_v3/per_frame_transforms.csv`
- `results/real_baseline/court_reg_v3/overlay.mp4`
- `results/real_baseline/court_reg_v3/feature_roi_mask.png`
- `results/real_baseline/court_reg_v1/line_params.csv`
- `results/real_baseline/court_reg_v1/overlay.mp4`
- demo event-frame PNGs referenced by `demo_foot_fault_events.csv`

If you need a perfectly self-consistent final-report archive, rerun the registration/demo scripts so these referenced outputs exist, then rebuild this bundle.

## Large Raw Assets Not Copied

The root raw videos/model weights are intentionally not duplicated into this condensed bundle because they are large and are not needed for writing the paper text:

- `IMG_8144.MOV`
- `IMG_8166.MOV`
- `yolov8n.pt`
- `yolov8x.pt`
- `models/yolov8n-pose.onnx`

They remain in the repo if a later model or script needs to reproduce detections.
