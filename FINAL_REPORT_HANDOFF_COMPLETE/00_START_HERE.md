# KitchenMaster Complete Final Report Handoff

This folder is the condensed, one-stop handoff for building the IEEE-style final report. It includes the current repo's source code, configs, tests, annotations, numeric results, review outputs, slide-ready images, generated charts, prior report notes, and a Claude-ready paste file.

## Read Order

1. `CLAUDE_FULL_HANDOFF.md` - paste this into Claude first.
2. `01_REPORT_WRITING_BRIEF.md` - report story, claims, limitations, and figure plan.
3. `02_RESULTS_DEEP_DIVE.md` - extracted metrics and comparisons with source paths.
4. `03_CODE_AND_EXPERIMENT_INDEX.md` - what every major file does and why it matters.
5. `04_REPRODUCIBILITY_AND_GAPS.md` - exact commands, test status, missing/generated artifacts, and what not to overclaim.
6. `05_IEEE_REPORT_MASTER_GUIDE.md` - final paper outline, best tables/figures, and strongest defensible claims.
7. `08_LATENCY_BENCHMARK_SUMMARY.md` - measured offline throughput for key pipeline variants.
8. `09_YOLO_LATENCY_RERUN_VARIANTS.md` - corrected YOLO timing variants, including non-tiled vs tiled CPU inference.
9. `10_REAL_WORLD_METRICS_GUIDE.md` - what real-world frame, event, latency, and accuracy metrics mean for the report.
10. `11_TIME_TO_DECISION_BENCHMARK_SUMMARY.md` - measured event-decision compute time and algorithmic delay.
11. `12_FULL_PIPELINE_VARIANT_BENCHMARK_SUMMARY.md` - repeated full-clip core pipeline timing across tracking variants.
12. `05_MASTER_ASSET_INDEX.csv` - paper-critical asset map with paths, hashes, and why each file matters.
13. `07_HANDOFF_VERIFICATION_SUMMARY.md` - verification that all paper-critical assets exist in this folder.
14. `manifests/` - copied files, skipped large files, and missing referenced outputs.

## Strongest Results

- 200 synthetic samples; legal P/R 1.0/0.94; fault P/R 0.5051/1.0; uncertain P/R 0.9444/0.51; uncertain rate 27.0%; false fault 0.0%; missed fault 0.0%.
- court_reg_v3 processed 2055 frames at 59.943 fps; 2055/2055 frames OK; 0 fallbacks; method anchor-point court model + ORB post_translation; left/right boundary edge strength means 12.14/7.31.
- demo_v1_active_side produced 3 events with labels {'foot_fault_volley': 1, 'uncertain': 2}, threshold 5.0 px and uncertainty margin 15.0 px.
- Latency benchmark: YOLOv8n CPU tiled ball tracking processed 300 frames at 0.694 fps, about 86.34x slower than the 59.943 fps source video; use this to support the offline/playback limitation.
- Time-to-decision benchmark: with active side treated as known/overridden, the foot-decision stage averaged about 556.35 ms per isolated event; this includes the YOLO-pose ONNX foot/person localizer (`models/yolov8n-pose.onnx`) but not the separate Ultralytics ball detector. The configured foot smoothing adds about 16.68 ms of future-frame algorithmic delay after the event frame is known.
- Full-pipeline variant benchmark: classical/blob core pipeline averaged 137.347 fps but only 35.67% ball detection coverage; YOLOv8n CPU non-tiled 640 averaged 28.951 fps and 14.16% coverage; YOLOv8n CPU non-tiled 1280 averaged 8.315 fps and 36.98% coverage. Automatic bounce discovery found 0 candidates in these runs, so manual demo event frames were used for foot-fault decisions.

## Important Folders

- `source_snapshot/` has code, configs, scripts, docs, requirements, and tests.
- `results/` has copied experiment outputs, review JSON, CSVs, debug frames, overlays, and result media under the size limit.
- `figures/photos_for_slides/` has the most report-ready images.
- `generated_charts/` and `tables/` provide report-friendly summaries generated from repo outputs.
- `data_annotations/` has real-video annotation JSON and reference frames.
- `prior_context/` and `prior_generated_bundle/` preserve earlier summaries and chat context.

## Large Assets

Large raw videos and model weights are intentionally listed but not duplicated. See `manifests/skipped_large_or_irrelevant_assets.csv`.
