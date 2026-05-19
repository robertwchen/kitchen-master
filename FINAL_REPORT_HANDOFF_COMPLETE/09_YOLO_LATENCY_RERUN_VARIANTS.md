# YOLO Variant Latency Rerun

This rerun separates normal YOLO inference from the repo demo configuration that used CPU tiled inference. Detector-only rows use a warmed-up model and preloaded frames; track rows include model load and tracking overhead.

| Case | Scope | Frames | Wall time (s) | FPS | x slower than 59.943 fps | Detected frames | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| yolov8n_cpu_notiled_imgsz640 | detector_only_warm_model | 60 | 1.854 | 32.359 | 1.85 | 63.3% | imgsz=640 tiled=False device=cpu tile_size=960 |
| yolov8n_cpu_notiled_imgsz1280 | detector_only_warm_model | 60 | 7.65 | 7.843 | 7.64 | 100.0% | imgsz=1280 tiled=False device=cpu tile_size=960 |
| yolov8n_cpu_tiled_imgsz1280_tile960 | detector_only_warm_model | 60 | 87.159 | 0.688 | 87.08 | 100.0% | imgsz=1280 tiled=True device=cpu tile_size=960 |
| track_yolov8n_cpu_notiled_imgsz640_60f | track_ball_end_to_end_including_model_load | 60 | 2.054 | 29.216 | 2.05 | 33.3% | imgsz=640 tiled=False device=cpu tile_size=960 |
| track_yolov8n_cpu_tiled_imgsz1280_60f | track_ball_end_to_end_including_model_load | 60 | 87.667 | 0.684 | 87.58 | 100.0% | imgsz=1280 tiled=True device=cpu tile_size=960 |

Interpretation: normal non-tiled YOLO is much faster than the original tiled CPU demo setting, but still below 59.943 fps on this machine. The original tiled setting is the reason the earlier 300-frame benchmark was extremely slow.
