# IEEE Final Report Master Guide

This folder is the one condensed package to give another model for writing the final IEEE-style report. It intentionally favors evidence you can defend in a short paper over raw volume.

## What To Paste Into Claude First

Use `CLAUDE_FULL_HANDOFF.md` as the first paste. Then upload or point Claude to this whole folder and tell it to use `05_IEEE_REPORT_MASTER_GUIDE.md`, `05_MASTER_ASSET_INDEX.csv`, and `07_HANDOFF_VERIFICATION_SUMMARY.md` as the asset map.

If Claude only has room for a few files, use this order:

1. `CLAUDE_FULL_HANDOFF.md`
2. `01_REPORT_WRITING_BRIEF.md`
3. `02_RESULTS_DEEP_DIVE.md`
4. `03_CODE_AND_EXPERIMENT_INDEX.md`
5. `04_REPRODUCIBILITY_AND_GAPS.md`
6. `05_MASTER_ASSET_INDEX.csv`

## Recommended 2-Page Paper Shape

Title idea: `KitchenMaster: Interpretable Side-View Pickleball Kitchen Fault Detection`

Abstract: State that the project studies whether a fixed side-view camera can support non-volley-zone foot-fault analysis. Mention the interpretable pipeline, anchor-based court registration, signed-distance labels, synthetic evaluation, and human-in-the-loop review. Avoid claiming a production referee.

Sections:

1. Introduction: NVZ foot faults are geometry-sensitive, and ambiguity near the line should not be forced into wrong confident calls.
2. System Overview: court registration, ball/event timing, foot localization, signed-distance decision, review artifacts.
3. Experiments: synthetic controlled data, real court registration versions/comparison, end-to-end demo pipeline.
4. Results: one compact metrics table, one registration comparison, one or two qualitative event figures.
5. Limitations and Future Work: real labeled event set is small, ball/event timing is fragile, foot localization remains review-dependent.

## Defensible Claims

- The project implements an interpretable research prototype for pickleball NVZ foot-fault analysis from a fixed side-view camera.
- The strongest real-video result is court registration: `court_reg_v3` processed `2055/2055` frames with `0` fallbacks using an anchor-point court model plus ORB `post_translation`.
- Registration comparison supports the `post_translation` choice: `post_translation` had `0` fallbacks while the affine comparison had `19`.
- Synthetic evaluation used `200` samples and achieved `0.0` false-fault rate and `0.0` missed-fault rate in the saved snapshot, with a `27.0%` uncertain rate.
- The demo pipeline is wired end-to-end and generated reviewable event outputs, but should be treated as qualitative/human-in-the-loop evidence.

## Claims To Avoid

- Do not claim autonomous officiating accuracy on real matches.
- Do not claim a large labeled real-video evaluation; the repo mainly supports registration evidence plus small demo-event outputs.
- Do not claim ball tracking is solved. The handoff notes that ball/event timing is the weakest stage.
- Do not cite missing optional files as if they exist; check `manifests/missing_referenced_files.csv`.

## Best Tables

Use `results/sim_v1/metrics.csv` for the main quantitative classification table:

- `n = 200`
- legal precision/recall: `1.0 / 0.94`
- fault precision/recall: `0.5051 / 1.0`
- uncertain precision/recall: `0.9444 / 0.51`
- uncertain rate: `0.27`
- false fault rate: `0.0`
- missed fault rate: `0.0`

Use `results/sim_v1/confusion_matrix.csv` only if there is room:

- true legal: `47 legal`, `0 fault`, `3 uncertain`
- true fault: `0 legal`, `50 fault`, `0 uncertain`
- true uncertain: `0 legal`, `49 fault`, `51 uncertain`

Use `results/real_baseline/court_reg_v3/comparisons/comparison_report.json` for the registration comparison:

- `post_translation_vs_affine_fixed`: `2055 OK / 0 fallback` vs `2036 OK / 19 fallback`
- `post_translation_vs_static`: both `2055 OK / 0 fallback`, but static does not model drift
- `refinement_on_vs_off`: affine with refinement on/off both `2036 OK / 19 fallback`

## Best Figures

For a 2-page report, use at most three figures:

- `figures/photos_for_slides/01_registration_overlay.png`: court geometry overlay.
- `figures/photos_for_slides/02_registration_comparison.png` or `generated_charts/registration_fallback_comparison.svg`: registration comparison.
- `figures/photos_for_slides/03_detected_fault_event.png`: detected event.

If discussing uncertainty, swap in one of:

- `figures/photos_for_slides/04_uncertain_event.png`
- `figures/photos_for_slides/05_uncertain_review_event.png`

## Methods Code To Cite

- `source_snapshot/src/court_model.py`: derives anchor-based NVZ geometry.
- `source_snapshot/src/stabilizer.py`: ORB matching, transform estimation, sanity checks, fallbacks.
- `source_snapshot/src/ball_tracker.py`: ball tracking and overlay output.
- `source_snapshot/src/volley_classifier.py`: trajectory/event candidates.
- `source_snapshot/src/foot_localizer.py`: foot localization strategies and review aids.
- `source_snapshot/src/foot_fault_pipeline.py`: signed-distance event labeling and review workflow.
- `source_snapshot/src/evaluate.py`: metrics and failure analysis.
- `source_snapshot/src/sim_generator.py`: controlled synthetic samples.

## Reproducibility Commands

The core commands are preserved in `04_REPRODUCIBILITY_AND_GAPS.md`:

```bash
python experiments/run_sim.py --config experiments/configs/sim_v1.yaml
python experiments/run_eval.py --results results/sim_v1/
python experiments/run_court_registration_v3.py --config experiments/configs/court_reg_v3.yaml
python experiments/run_demo_pipeline.py --config experiments/configs/demo_pipeline.yaml
python experiments/run_demo_pipeline.py --config experiments/configs/demo_pipeline.yaml --mode apply_overrides
pytest tests/
```

The handoff says `.venv/bin/python -m pytest tests/` previously passed with `27 passed`.

## Where Prior Prompts Matter

The prior chats in `prior_context/TotalPrompts/` are useful for framing, not for primary evidence. They document how the project narrowed from a broad idea into a fixed side-view, interpretable, uncertainty-aware research prototype. Treat metrics from saved result files as stronger evidence than numbers mentioned in early chats.

## Final Report Framing

The best final-report argument is:

KitchenMaster shows that a fixed side-view camera can support an interpretable NVZ foot-fault analysis pipeline when court geometry is registered carefully. The current system provides strong registration evidence and reviewable end-to-end outputs, while the final fault decision remains limited by ball/event timing, foot localization, and the lack of a larger labeled real-event dataset.

