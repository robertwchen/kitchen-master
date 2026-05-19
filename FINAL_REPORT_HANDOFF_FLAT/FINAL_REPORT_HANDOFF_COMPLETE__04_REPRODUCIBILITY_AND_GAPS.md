# Reproducibility And Gaps

## Main Commands

```bash
python experiments/run_sim.py --config experiments/configs/sim_v1.yaml
python experiments/run_eval.py --results results/sim_v1/
python experiments/run_court_registration_v3.py --config experiments/configs/court_reg_v3.yaml
python experiments/run_demo_pipeline.py --config experiments/configs/demo_pipeline.yaml
python experiments/run_demo_pipeline.py --config experiments/configs/demo_pipeline.yaml --mode apply_overrides
pytest tests/
```

## Test Status

Verified in the repo virtual environment with `.venv/bin/python -m pytest tests/`: `27 passed`.

## Important Gaps / Honest Limitations

- The project has strong real-video court registration evidence, but limited fully labeled real-video foot-fault evaluation.
- The demo pipeline is human-in-the-loop. It produces review artifacts and supports overrides; it should not be framed as fully autonomous.
- Ball/event timing is the weakest stage. The active-side demo snapshot reports a much lower detection rate than the base demo snapshot, and no automatic bounce candidates were found in those pending review files.
- Some files are referenced by reports/configs but may be missing or intentionally not copied if huge. Check `manifests/missing_referenced_files.csv` and `manifests/skipped_large_or_irrelevant_assets.csv`.
- Large raw videos and model weights remain in the repo root/models folder and are listed in manifests rather than duplicated.

## Best Next Reruns If Time Allows

1. Rerun `court_reg_v3` if you need all per-frame CSV/video outputs regenerated cleanly.
2. Rerun `run_demo_pipeline.py` after preparing `review_approved.json` to produce finalized override-based outputs.
3. Run `pytest tests/` inside the `.venv` or install pytest if unavailable.
4. Add a small labeled real-event table if a stronger final-report quantitative claim is needed.
