# Real-World Metrics Guide For Final Report

This note translates the repo outputs and latency benchmarks into report-safe performance language. Use it to avoid confusing video frame spacing, model latency, throughput, and event-decision latency.

## What Real-Time Would Require

The main real clip is `59.943 fps`, so frames arrive every:

- `1 / 59.943 = 0.0167 s`
- about `16.7 ms per frame`

For a true live 60 fps system, the whole per-frame pipeline must average below `16.7 ms/frame`, and the p95/p99 latency should also stay near that budget. Average FPS alone is not enough because occasional slow frames can delay alerts.

For a 30 fps deployment target, the budget is about `33.3 ms/frame`.

## Measured Offline Throughput In This Repo

Measured on this machine with wall-clock timing. These are offline throughput numbers, not deployment guarantees.

| Component / variant | Measured throughput | Approx. ms/frame | Real-time interpretation |
|---|---:|---:|---|
| Court registration v3, post-translation | `129.765 fps` | `7.7 ms/frame` | Faster than 60 fps in isolation |
| Classical/blob ball tracking | `117.591 fps` | `8.5 ms/frame` | Faster than 60 fps in isolation |
| YOLOv8n CPU non-tiled `imgsz=640`, through `track_ball` | `29.216 fps` | `34.2 ms/frame` | Roughly 2x slower than 60 fps, near 30 fps only in isolation |
| YOLOv8n CPU non-tiled `imgsz=1280`, detector only | `7.843 fps` | `127.5 ms/frame` | Too slow for live 60 fps |
| YOLOv8n CPU tiled `imgsz=1280`, through `track_ball` | `0.684 fps` | `1462 ms/frame` | Playback/offline only |
| Court registration v1 | `23.091 fps` | `43.3 ms/frame` | Slower than 60 fps and less reliable geometrically |

Important interpretation: YOLO is not inherently `0.684 fps`; the slow result comes from the original high-resolution tiled CPU demo setting. Normal non-tiled YOLO is much faster, but still not fast enough for live 60 fps on this machine.

## Metrics Needed For Per-Frame Detection Claims

To claim real-time frame-level detection, the report would need:

- `mean_latency_ms_per_frame`: average processing time per frame.
- `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`: tail latency, because live systems fail when occasional frames are too slow.
- `effective_fps`: processed frames divided by wall-clock time.
- `dropped_frame_rate`: percent of frames skipped or processed late.
- `memory_usage` and device used: CPU, GPU, Apple MPS, etc.
- `input_resolution` and model config: for example `imgsz=640` vs `imgsz=1280`, tiled vs non-tiled.

For this project, the most honest statement is that these were benchmarked offline for selected components, not as a deployed streaming system.

## Metrics Needed For Ball Detection

A real-world ball detector should be evaluated with labeled ball positions or bounding boxes:

- `ball_detection_rate`: percent of frames where the ball is detected.
- `ball_precision`: detected balls that are actually the ball.
- `ball_recall`: labeled ball frames recovered by the detector.
- `mean_ball_center_error_px`: distance between predicted and labeled ball center.
- `track_fragmentation`: number of broken track segments.
- `max_gap_frames` and `mean_gap_frames`: how often tracking drops out.
- `event_window_detection_rate`: whether the ball is detected near hits/bounces, not just anywhere.

Current repo evidence includes detection rates in review files, such as `94.5%` for one demo run and `35.7%` for the active-side run, but it does not include a large hand-labeled ball benchmark.

## Metrics Needed For Court Registration

Court geometry is the strongest current result. For real-world validation, use:

- `registration_success_rate`: frames with valid court model.
- `fallback_rate`: frames where the transform failed and fallback was used.
- `reprojection_error_px`: distance from predicted court anchors to manually labeled anchors on sampled frames.
- `line_error_px`: distance between predicted NVZ boundary and manually labeled line.
- `edge_strength`: how well predicted boundary aligns with image edges.

Current repo evidence:

- `court_reg_v3`: `2055/2055` frames registered, `0` fallbacks.
- `post_translation` comparison: `0` fallbacks versus affine comparison with `19` fallbacks.
- Reprojection labels were not fully populated, so do not claim centimeter-level ground truth accuracy.

## Metrics Needed For Foot Localization

For real-world foot-fault detection, the foot location must be evaluated against labeled contact points:

- `foot_contact_error_px`: distance between predicted foot contact point and human-labeled contact point.
- `foot_detection_rate`: percent of event frames where a usable foot point is produced.
- `low_confidence_rate`: percent marked low confidence or requiring review.
- `wrong_player_rate`: percent where the wrong side/player was selected.
- `active_side_accuracy`: whether the system selects the player who hit the ball.
- `distance_error_px`: error in signed distance from foot to NVZ boundary.

Current repo evidence is qualitative/event-level. It has reviewable foot localization outputs for 3 demo events, but not a large labeled foot-contact benchmark.

## Metrics Needed For Hit / Volley / Foot-Fault Events

For final foot-fault claims, the most important unit is the event, not every frame. A real benchmark should label actual hit/volley frames and whether each is legal/fault/uncertain.

Needed metrics:

- `event_precision`: predicted foot-fault events that correspond to real labeled events.
- `event_recall`: real labeled events detected by the system.
- `event_timing_error_frames`: predicted event frame minus labeled event frame.
- `event_timing_error_ms`: timing error in milliseconds.
- `false_fault_rate`: legal events incorrectly called faults.
- `missed_fault_rate`: true foot faults missed by the system.
- `uncertain_rate`: events deferred to review.
- `review_required_rate`: events that cannot be confidently automated.
- `time_to_decision_ms`: wall-clock time from the hit frame to the system output.

Useful timing conversions at `59.943 fps`:

- `1 frame` = `16.7 ms`
- `3 frames` = `50.0 ms`
- `5 frames` = `83.4 ms`
- `12 frames` = `200.2 ms`
- `30 frames` = `500.5 ms`

Current repo evidence:

- Synthetic baseline: `0.0%` false fault, `0.0%` missed fault, `27.0%` uncertain rate on 200 synthetic samples.
- Demo active-side run: 3 events, labels `{foot_fault_volley: 1, uncertain: 2}`.
- Demo review is human-in-the-loop, so do not report it as autonomous real-match accuracy.

## Measured Time-To-Decision Snapshot

See `11_TIME_TO_DECISION_BENCHMARK_SUMMARY.md` for the latest benchmark.

Measured with active side treated as known/overridden to isolate the foot-decision stage:

- Median 3-event decision run: `1.5773 s`.
- Mean compute per event across 3-event runs: `533.78 ms/event`.
- Mean isolated compute per event: `556.35 ms/event`.
- Event 929: `538.36 ms`, label `foot_fault_volley` with right-side override.
- Event 1537: `612.11 ms`, label `uncertain` with right-side override.
- Event 1948: `518.58 ms`, label `legal_volley` with left-side override.

Important caveat: because active side was overridden for this benchmark, these labels should not replace the autonomous demo result. The autonomous active-side demo still produced `{foot_fault_volley: 1, uncertain: 2}` because frame 1948 had active-side ambiguity.

This foot-decision timing does include the YOLO-pose foot/person cue used by `event_hybrid`: `src/foot_localizer.py` loads `models/yolov8n-pose.onnx` through OpenCV DNN, selects a person/leg near the NVZ boundary, builds a lower-body ROI from pose keypoints, then refines the foot contact point with motion/threshold/edge masks and temporal smoothing. It does not include the separate Ultralytics `yolov8n.pt` ball detector.

## Repeated Full-Pipeline Variant Snapshot

See `12_FULL_PIPELINE_VARIANT_BENCHMARK_SUMMARY.md` for repeated full-clip core pipeline runs.

This benchmark measured:

`track_ball` + `run_volley_classification` + manual demo event construction + `run_foot_fault_pipeline`

The registration CSV was precomputed from `court_reg_v3`, matching the normal demo pipeline. Overlay/debug video writing was disabled to focus on processing throughput.

Aggregated results over the 2055-frame real clip:

- Classical/blob tracking: 3 reps, `14.9685 s` mean total time, `137.347 fps` mean effective throughput, `35.67%` ball detection coverage, labels `{"foot_fault_volley": 1, "uncertain": 2}`.
- YOLOv8n CPU non-tiled `imgsz=640`: 3 reps, `71.1139 s` mean total time, `28.951 fps`, `14.16%` ball detection coverage, labels `{"uncertain": 3}`.
- YOLOv8n CPU non-tiled `imgsz=1280`: 1 rep, `247.1482 s` total time, `8.315 fps`, `36.98%` ball detection coverage, labels `{"legal_volley": 2, "uncertain": 1}`.
- YOLOv8n CPU tiled `imgsz=1280`: not rerun over full clip; prior 300-frame measurement at `0.694 fps` implies about `49.4 min` for ball tracking alone over 2055 frames.

Important caveat: automatic bounce/event discovery produced `0` bounce candidates in these runs, so the benchmark used the same manual demo event frames for foot-fault decisions. This is the right number for core pipeline playback throughput, but not a proof of fully autonomous event detection.

## Algorithmic Delay Versus Compute Latency

There are two different delays:

1. Compute latency: how long the software takes to process frames.
2. Algorithmic delay: how many future frames the algorithm waits for before making a decision.

Examples from the config:

- Bounce logic uses `lookahead_frames=5`, so it may require about `83 ms` of future video before confirming a bounce-like trajectory.
- Foot localization uses `temporal_window_radius=1`, so it looks about `16.7 ms` before and after the event frame.
- Active-side inference uses `active_side_window_frames=12`, about `200 ms` of context.
- Hit classification uses `hit_lookback_frames=30`, about `500 ms` of past context.

Even if compute were instant, event decisions can still have algorithmic delay because the system uses temporal context.

## Report-Safe Limitation Paragraph

Use language like this:

> The current prototype is best understood as an offline playback and review pipeline rather than a deployed real-time detector. In component benchmarks, court registration v3 and the classical ball tracker ran faster than the 59.943 fps source video, but the learned YOLO ball detector was configuration-dependent: a normal non-tiled YOLOv8n CPU configuration reached 29.2 fps through the ball-tracking stage, while the high-resolution tiled CPU configuration used in the demo reached only 0.684 fps. Since reliable real-time use would require the full pipeline, including ball detection, event timing, foot localization, and decision logic, to stay below roughly 16.7 ms per frame at 60 fps, the current system is not yet a live referee. Future work should benchmark p95/p99 end-to-end latency on GPU hardware and evaluate event-level precision, recall, timing error, and false-fault/missed-fault rates on labeled real gameplay.

## Best Metric Set For A Final Report Table

If space allows, include one small table with:

- `court_reg_v3`: `2055 frames`, `0 fallbacks`, `129.765 fps measured offline`.
- `classical ball tracker`: `117.591 fps`, detection rate from relevant run if cited.
- `YOLOv8n CPU non-tiled imgsz=640`: `29.216 fps` through `track_ball`.
- `YOLOv8n CPU tiled imgsz=1280`: `0.684 fps` through `track_ball`.
- `demo active-side`: `3 events`, `1 foot_fault_volley`, `2 uncertain`.
- `synthetic baseline`: `200 samples`, `0.0% false fault`, `0.0% missed fault`, `27.0% uncertain`.

