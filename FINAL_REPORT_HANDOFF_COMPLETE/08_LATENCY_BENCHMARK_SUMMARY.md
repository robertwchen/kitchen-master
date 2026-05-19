# Final Report Latency Benchmark Summary

Measured with wall-clock `time.perf_counter()` on this machine. These are offline processing timings, not real-time guarantees.

| Case | Wall time (s) | Frames/samples | Effective FPS | Source FPS | Slower than real time | Notes |
|---|---:|---:|---:|---:|---:|---|
| synthetic_sim_v1_full_200_samples | 18.652 | 200 | 10.723 |  |  | 200 synthetic images, classical detector/evaluation/plots/overlays |
| court_registration_v1_full_clip | 109.782 | 2535 | 23.091 | 59.943 | 2.6 | Static line model; writes per-frame CSV, debug frames, overlay at 10 fps |
| court_registration_v3_post_translation_full_clip | 15.836 | 2055 | 129.765 | 59.943 | 0.46 | Anchor court model + post_translation; comparison exports disabled for primary-method timing |
| ball_tracking_blob_classical_300_frames | 2.551 | 300 | 117.591 | 59.943 | 0.51 | Classical diff/HSV blob tracker sample; overlay/debug disabled |
| ball_tracking_yolov8n_cpu_tiled_300_frames | 432.123 | 300 | 0.694 | 59.943 | 86.34 | YOLOv8n COCO sports-ball detector on CPU, imgsz=1280, tiled inference; overlay/debug disabled |

Report-safe interpretation: the saved pipeline is offline/review-oriented. The YOLO CPU ball-tracking sample is the strongest evidence that the current learned-detector path is not real-time on this setup.

## Correction / YOLO Variant Rerun

The very slow YOLO result above is real for the original demo setting (`device=cpu`, `imgsz=1280`, `use_tiled_inference=true`), but it should not be generalized to all YOLO use. A follow-up benchmark in `09_YOLO_LATENCY_RERUN_VARIANTS.md` found:

- YOLOv8n CPU, non-tiled, `imgsz=640`: 32.359 fps detector-only with warm model; 29.216 fps through `track_ball` including model load, about 2.05x slower than the 59.943 fps source video.
- YOLOv8n CPU, non-tiled, `imgsz=1280`: 7.843 fps detector-only, about 7.64x slower than source video.
- YOLOv8n CPU, tiled, `imgsz=1280`, `tile_size=960`: 0.688 fps detector-only and 0.684 fps through `track_ball`, about 87x slower than source video.

Report-safe interpretation: the current repo demo configuration was far too slow because it used tiled CPU inference, while normal non-tiled YOLO is much faster but still not real-time at 60 fps on this machine. The limitation should be framed as configuration/hardware dependent, not as an inherent YOLO limitation.
