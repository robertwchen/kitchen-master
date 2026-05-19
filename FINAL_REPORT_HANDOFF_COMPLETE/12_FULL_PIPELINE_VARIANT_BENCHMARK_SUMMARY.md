# Full Pipeline Variant Benchmark

This benchmark measures the core offline pipeline over the real 2055-frame clip: `track_ball` + `run_volley_classification` + manual demo event construction + `run_foot_fault_pipeline`. Because automatic bounce/event discovery produced zero bounce candidates in these runs, the event frames were the same manual demo frames used by the active-side demo. Overlay/debug video writing was disabled to focus on processing throughput. The registration CSV was precomputed, matching the normal demo pipeline that consumes `court_reg_v3` outputs. This is therefore a core pipeline throughput benchmark with manual event frames, not proof of fully autonomous event discovery.

## Aggregated Full-Clip Results

| Variant | Reps | Mean total time (s) | Median total time (s) | Mean effective FPS | x slower than 59.943 fps | Ball tracking FPS | Ball detection rate | Last labels |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| blob_classical | 3 | 14.9685 | 14.8747 | 137.347 | 0.44 | 155.157 | 35.67% | `{"foot_fault_volley": 1, "uncertain": 2}` |
| yolo1280_notiled_cpu | 1 | 247.1482 | 247.1482 | 8.315 | 7.21 | 8.433 | 36.98% | `{"legal_volley": 2, "uncertain": 1}` |
| yolo640_notiled_cpu | 3 | 71.1139 | 69.4406 | 28.951 | 2.08 | 30.425 | 14.16% | `{"uncertain": 3}` |

## Slow Tiled Reference

The original high-resolution tiled CPU YOLO setting was not rerun over the full 2055-frame clip because the prior measured 300-frame benchmark took 432.123 s at 0.694 fps. Extrapolated to 2055 frames, ball tracking alone would take about 49.4 minutes, before event and foot-decision stages.

## Report-Safe Interpretation

For the real clip, the classical/blob core pipeline is faster than real time but has low ball detection coverage, and all variants found zero automatic bounce candidates in this configuration. The non-tiled YOLOv8n `imgsz=640` pipeline is much more practical than tiled inference but remains below the 59.943 fps source rate on CPU. Higher-resolution YOLO improves frame-level detection coverage in these samples but is far slower. These measurements support the claim that the current system is an offline playback/review prototype unless model configuration and hardware acceleration are improved.
