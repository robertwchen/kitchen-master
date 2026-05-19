
import csv
import copy
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp' / 'final_report_latency_benchmarks'
CFG_DIR = OUT / 'configs'
RESULTS_DIR = OUT / 'results'
CFG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PYTHON = ROOT / '.venv' / 'bin' / 'python'
rows = []

def load_yaml(rel):
    with open(ROOT / rel) as f:
        return yaml.safe_load(f)

def write_yaml(name, cfg):
    path = CFG_DIR / name
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path

def run_case(name, cmd, frames=None, source_fps=None, notes=''):
    print(f'\n=== {name} ===', flush=True)
    print(' '.join(str(x) for x in cmd), flush=True)
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    print(proc.stdout[-4000:], flush=True)
    eff_fps = (frames / elapsed) if frames else None
    realtime_ratio = (source_fps / eff_fps) if (source_fps and eff_fps and eff_fps > 0) else None
    rows.append({
        'case': name,
        'exit_code': proc.returncode,
        'wall_time_s': round(elapsed, 3),
        'frames_or_samples': frames if frames is not None else '',
        'effective_fps_or_samples_per_s': round(eff_fps, 3) if eff_fps is not None else '',
        'source_fps': source_fps if source_fps is not None else '',
        'times_slower_than_realtime': round(realtime_ratio, 2) if realtime_ratio is not None else '',
        'notes': notes,
    })
    if proc.returncode != 0:
        print(f'WARNING: {name} failed with exit code {proc.returncode}', flush=True)
    return proc.returncode

# 1. Synthetic full 200-sample pipeline.
sim = load_yaml('experiments/configs/sim_v1.yaml')
sim['run_name'] = 'latency_sim_v1'
sim['output']['results_dir'] = str(RESULTS_DIR)
sim_cfg = write_yaml('sim_v1_latency.yaml', sim)
run_case(
    'synthetic_sim_v1_full_200_samples',
    [str(PYTHON), 'experiments/run_sim.py', '--config', str(sim_cfg)],
    frames=200,
    notes='200 synthetic images, classical detector/evaluation/plots/overlays',
)

# 2. Court registration v1 full clip.
v1 = load_yaml('experiments/configs/court_reg_v1.yaml')
v1['run_name'] = 'latency_court_reg_v1'
v1['output']['results_dir'] = str(RESULTS_DIR)
v1_cfg = write_yaml('court_reg_v1_latency.yaml', v1)
run_case(
    'court_registration_v1_full_clip',
    [str(PYTHON), 'experiments/run_court_registration.py', '--config', str(v1_cfg)],
    frames=2535,
    source_fps=59.943,
    notes='Static line model; writes per-frame CSV, debug frames, overlay at 10 fps',
)

# 3. Court registration v3 full clip. Disable comparison exports to measure primary method, but keep overlay/debug like report config.
v3 = load_yaml('experiments/configs/court_reg_v3.yaml')
v3['run_name'] = 'latency_court_reg_v3_post_translation'
v3['output']['results_dir'] = str(RESULTS_DIR)
v3['comparison_exports']['enabled'] = False
v3_cfg = write_yaml('court_reg_v3_latency.yaml', v3)
run_case(
    'court_registration_v3_post_translation_full_clip',
    [str(PYTHON), 'experiments/run_court_registration_v3.py', '--config', str(v3_cfg)],
    frames=2055,
    source_fps=59.943,
    notes='Anchor court model + post_translation; comparison exports disabled for primary-method timing',
)

# Ball tracker samples: first 300 frames, no overlay/debug to focus on processing speed.
base_demo = load_yaml('experiments/configs/demo_pipeline.yaml')
base_bt = copy.deepcopy(base_demo['ball_tracking'])
base_bt['write_overlay'] = False
base_bt['debug_every_n'] = 999999
video_path = ROOT / base_demo['video']['path']

def write_ball_cfg(name, bt_cfg):
    cfg = {
        'video_path': str(video_path),
        'output_dir': str(RESULTS_DIR / name),
        'clip_start_frame': 0,
        'clip_end_frame': 300,
        'ball_tracking': bt_cfg,
    }
    path = CFG_DIR / f'{name}.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path

ball_runner = OUT / 'run_ball_tracking_case.py'
ball_runner.write_text("""
import sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ball_tracker import track_ball
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
rows = track_ball(
    video_path=Path(cfg['video_path']),
    output_dir=Path(cfg['output_dir']),
    cfg=cfg['ball_tracking'],
    clip_start_frame=int(cfg['clip_start_frame']),
    clip_end_frame=int(cfg['clip_end_frame']),
    debug_every_n=int(cfg['ball_tracking'].get('debug_every_n', 999999)),
    write_overlay=bool(cfg['ball_tracking'].get('write_overlay', False)),
    overlay_fps=float(cfg['ball_tracking'].get('overlay_fps', 10.0)),
    overlay_scale=float(cfg['ball_tracking'].get('overlay_scale', 0.5)),
)
print(f'rows={len(rows)}')
if rows:
    detected=sum(1 for r in rows if r.get('ball_x') not in (None, ''))
    print(f'detected={detected} detection_rate_pct={100*detected/len(rows):.1f}')
""")

blob = copy.deepcopy(base_bt)
blob['tracking_backend'] = 'blob'
blob_cfg = write_ball_cfg('ball_tracking_blob_300_frames', blob)
run_case(
    'ball_tracking_blob_classical_300_frames',
    [str(PYTHON), str(ball_runner), str(blob_cfg)],
    frames=300,
    source_fps=59.943,
    notes='Classical diff/HSV blob tracker sample; overlay/debug disabled',
)

yolo = copy.deepcopy(base_bt)
yolo['tracking_backend'] = 'ultralytics'
yolo_cfg = write_ball_cfg('ball_tracking_yolo_cpu_300_frames', yolo)
run_case(
    'ball_tracking_yolov8n_cpu_tiled_300_frames',
    [str(PYTHON), str(ball_runner), str(yolo_cfg)],
    frames=300,
    source_fps=59.943,
    notes='YOLOv8n COCO sports-ball detector on CPU, imgsz=1280, tiled inference; overlay/debug disabled',
)

csv_path = OUT / 'latency_benchmark_results.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

md_path = OUT / 'latency_benchmark_summary.md'
lines = ['# Final Report Latency Benchmark Summary', '', 'Measured with wall-clock `time.perf_counter()` on this machine. These are offline processing timings, not real-time guarantees.', '', '| Case | Wall time (s) | Frames/samples | Effective FPS | Source FPS | Slower than real time | Notes |', '|---|---:|---:|---:|---:|---:|---|']
for r in rows:
    lines.append(f"| {r['case']} | {r['wall_time_s']} | {r['frames_or_samples']} | {r['effective_fps_or_samples_per_s']} | {r['source_fps']} | {r['times_slower_than_realtime']} | {r['notes']} |")
lines += ['', 'Report-safe interpretation: the saved pipeline is offline/review-oriented. The YOLO CPU ball-tracking sample is the strongest evidence that the current learned-detector path is not real-time on this setup.']
md_path.write_text('\n'.join(lines) + '\n')
print(f'\nWROTE {csv_path}')
print(f'WROTE {md_path}')
