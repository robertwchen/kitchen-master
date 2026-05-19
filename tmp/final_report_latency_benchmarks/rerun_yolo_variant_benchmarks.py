
import csv
import copy
import time
from pathlib import Path

import cv2
import yaml

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ball_detector import UltralyticsBallDetector
from src.ball_tracker import track_ball

OUT = ROOT / 'tmp' / 'final_report_latency_benchmarks'
RESULTS_DIR = OUT / 'rerun_yolo_variants'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / 'experiments/configs/demo_pipeline.yaml') as f:
    demo = yaml.safe_load(f)
base_bt = copy.deepcopy(demo['ball_tracking'])
video_path = ROOT / demo['video']['path']
source_fps = 59.943
n_frames = 60
warmup_frames = 5

# Load frames once so disk reading is not the differentiator for detector-only tests.
cap = cv2.VideoCapture(str(video_path))
frames = []
while len(frames) < n_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f'loaded_frames={len(frames)}')

rows = []

def make_cfg(imgsz=640, tiled=False, device='cpu', tile_size=960, overlap=0.25, max_det=8):
    cfg = copy.deepcopy(base_bt)
    cfg['tracking_backend'] = 'ultralytics'
    cfg['write_overlay'] = False
    cfg['debug_every_n'] = 999999
    cfg['ultralytics']['imgsz'] = imgsz
    cfg['ultralytics']['use_tiled_inference'] = tiled
    cfg['ultralytics']['device'] = device
    cfg['ultralytics']['tile_size'] = tile_size
    cfg['ultralytics']['tile_overlap'] = overlap
    cfg['ultralytics']['max_det'] = max_det
    return cfg

def detector_case(name, cfg):
    print(f'\n=== detector {name} ===', flush=True)
    detector = UltralyticsBallDetector.from_config(cfg)
    # Warmup a few predictions.
    for frame in frames[:warmup_frames]:
        detector.detect(frame)
    start = time.perf_counter()
    detections = 0
    for frame in frames:
        cands = detector.detect(frame)
        if cands:
            detections += 1
    elapsed = time.perf_counter() - start
    fps = len(frames) / elapsed
    rows.append({
        'case': name,
        'scope': 'detector_only_warm_model',
        'frames': len(frames),
        'wall_time_s': round(elapsed, 3),
        'effective_fps': round(fps, 3),
        'times_slower_than_59_943fps': round(source_fps / fps, 2),
        'detected_frame_pct': round(100 * detections / len(frames), 1),
        'notes': f"imgsz={cfg['ultralytics']['imgsz']} tiled={cfg['ultralytics']['use_tiled_inference']} device={cfg['ultralytics']['device']} tile_size={cfg['ultralytics']['tile_size']}",
    })
    print(rows[-1], flush=True)

def track_case(name, cfg):
    print(f'\n=== track_ball {name} ===', flush=True)
    start = time.perf_counter()
    out_dir = RESULTS_DIR / name
    rows_out = track_ball(
        video_path=video_path,
        output_dir=out_dir,
        cfg=cfg,
        clip_start_frame=0,
        clip_end_frame=n_frames,
        debug_every_n=999999,
        write_overlay=False,
        overlay_fps=10.0,
        overlay_scale=0.5,
    )
    elapsed = time.perf_counter() - start
    fps = n_frames / elapsed
    detected = sum(1 for r in rows_out if r.get('ball_x') is not None)
    rows.append({
        'case': name,
        'scope': 'track_ball_end_to_end_including_model_load',
        'frames': n_frames,
        'wall_time_s': round(elapsed, 3),
        'effective_fps': round(fps, 3),
        'times_slower_than_59_943fps': round(source_fps / fps, 2),
        'detected_frame_pct': round(100 * detected / n_frames, 1),
        'notes': f"imgsz={cfg['ultralytics']['imgsz']} tiled={cfg['ultralytics']['use_tiled_inference']} device={cfg['ultralytics']['device']} tile_size={cfg['ultralytics']['tile_size']}",
    })
    print(rows[-1], flush=True)

variants = [
    ('yolov8n_cpu_notiled_imgsz640', make_cfg(imgsz=640, tiled=False, device='cpu')),
    ('yolov8n_cpu_notiled_imgsz1280', make_cfg(imgsz=1280, tiled=False, device='cpu')),
    ('yolov8n_cpu_tiled_imgsz1280_tile960', make_cfg(imgsz=1280, tiled=True, device='cpu', tile_size=960, overlap=0.25)),
]

for name, cfg in variants:
    detector_case(name, cfg)

# Run end-to-end track_ball on the fast/normal and original slow configs for apples-to-apples pipeline cost.
track_case('track_yolov8n_cpu_notiled_imgsz640_60f', variants[0][1])
track_case('track_yolov8n_cpu_tiled_imgsz1280_60f', variants[2][1])

csv_path = OUT / 'latency_benchmark_yolo_rerun_variants.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

md_path = OUT / 'latency_benchmark_yolo_rerun_variants.md'
lines = ['# YOLO Variant Latency Rerun', '', 'This rerun separates normal YOLO inference from the repo demo configuration that used CPU tiled inference. Detector-only rows use a warmed-up model and preloaded frames; track rows include model load and tracking overhead.', '', '| Case | Scope | Frames | Wall time (s) | FPS | x slower than 59.943 fps | Detected frames | Notes |', '|---|---|---:|---:|---:|---:|---:|---|']
for r in rows:
    lines.append(f"| {r['case']} | {r['scope']} | {r['frames']} | {r['wall_time_s']} | {r['effective_fps']} | {r['times_slower_than_59_943fps']} | {r['detected_frame_pct']}% | {r['notes']} |")
lines.append('')
lines.append('Interpretation: normal non-tiled YOLO is much faster than the original tiled CPU demo setting, but still below 59.943 fps on this machine. The original tiled setting is the reason the earlier 300-frame benchmark was extremely slow.')
md_path.write_text('\n'.join(lines) + '\n')
print(f'WROTE {csv_path}')
print(f'WROTE {md_path}')
