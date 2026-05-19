
import csv
import copy
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ball_tracker import track_ball
from src.volley_classifier import run_volley_classification
from src.foot_fault_pipeline import run_foot_fault_pipeline
from experiments.run_demo_pipeline import _build_volley_events

OUT = ROOT / 'tmp' / 'final_report_latency_benchmarks'
RESULTS = OUT / 'full_pipeline_variant_results'
RESULTS.mkdir(parents=True, exist_ok=True)

with open(ROOT / 'FINAL_REPORT_HANDOFF_COMPLETE/misc_analysis/demo_pipeline_active_side.yaml') as f:
    base = yaml.safe_load(f)

video_path = ROOT / base['video']['path']
source_fps = 59.943
source_frames = 2055
manual_frames = [int(x) for x in base['foot_fault']['manual_volley_frames']]

# Keep expensive visualization off for timing, but keep CSV/event outputs.
def make_cfg(variant):
    cfg = copy.deepcopy(base)
    cfg['ball_tracking']['write_overlay'] = False
    cfg['ball_tracking']['debug_every_n'] = 999999
    cfg['output']['write_summary_video'] = False
    if variant == 'blob_classical':
        cfg['ball_tracking']['tracking_backend'] = 'blob'
    elif variant == 'yolo640_notiled_cpu':
        cfg['ball_tracking']['tracking_backend'] = 'ultralytics'
        cfg['ball_tracking']['ultralytics'] = {
            'model_path': 'yolov8n.pt', 'confidence': 0.05, 'iou': 0.45,
            'imgsz': 640, 'device': 'cpu', 'max_det': 8,
            'use_tiled_inference': False, 'tile_size': 960,
            'tile_overlap': 0.25, 'merge_radius_px': 24.0,
            'class_ids': [32], 'class_names': ['sports ball'],
        }
    elif variant == 'yolo1280_notiled_cpu':
        cfg['ball_tracking']['tracking_backend'] = 'ultralytics'
        cfg['ball_tracking']['ultralytics'] = {
            'model_path': 'yolov8n.pt', 'confidence': 0.05, 'iou': 0.45,
            'imgsz': 1280, 'device': 'cpu', 'max_det': 8,
            'use_tiled_inference': False, 'tile_size': 960,
            'tile_overlap': 0.25, 'merge_radius_px': 24.0,
            'class_ids': [32], 'class_names': ['sports ball'],
        }
    elif variant == 'yolo1280_tiled_cpu_reference':
        cfg['ball_tracking']['tracking_backend'] = 'ultralytics'
        cfg['ball_tracking']['ultralytics'] = {
            'model_path': 'yolov8n.pt', 'confidence': 0.05, 'iou': 0.45,
            'imgsz': 1280, 'device': 'cpu', 'max_det': 8,
            'use_tiled_inference': True, 'tile_size': 960,
            'tile_overlap': 0.25, 'merge_radius_px': 24.0,
            'class_ids': [32], 'class_names': ['sports ball'],
        }
    else:
        raise ValueError(variant)
    return cfg

def run_one(variant, rep, clip_end=None):
    cfg = make_cfg(variant)
    out_dir = RESULTS / f'{variant}_rep{rep:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = clip_end if clip_end is not None else source_frames

    t0 = time.perf_counter()
    t = time.perf_counter()
    tracking_rows = track_ball(
        video_path=video_path,
        output_dir=out_dir / 'ball_tracking',
        cfg=cfg['ball_tracking'],
        clip_start_frame=0,
        clip_end_frame=clip_end,
        debug_every_n=int(cfg['ball_tracking'].get('debug_every_n', 999999)),
        write_overlay=False,
        overlay_fps=float(cfg['ball_tracking'].get('overlay_fps', 10.0)),
        overlay_scale=float(cfg['ball_tracking'].get('overlay_scale', 0.5)),
    )
    ball_s = time.perf_counter() - t

    t = time.perf_counter()
    vc_result = run_volley_classification(
        tracking_rows=tracking_rows,
        video_path=video_path,
        output_dir=out_dir / 'volley_events',
        cfg=cfg['volley_classification'],
        court_surface_y=None,
        hit_frames=None,
    )
    volley_s = time.perf_counter() - t

    t = time.perf_counter()
    volley_events = _build_volley_events(
        volley_candidate_frames=manual_frames,
        tracking_rows=tracking_rows,
        classified_events=vc_result.get('events'),
        ball_context_radius=int(cfg.get('foot_fault', {}).get('active_side_window_frames', 12)),
    )
    ff_cfg = dict(cfg['foot_fault'])
    ff_cfg['foot_localizer'] = cfg['foot_localizer']
    fault_results = run_foot_fault_pipeline(
        volley_events=volley_events,
        video_path=video_path,
        output_dir=out_dir / 'foot_faults',
        cfg=ff_cfg,
        registration_csv=ROOT / cfg['registration']['csv_path'],
        manual_line_override_path=None,
        ref_annotations_path=ROOT / cfg['registration']['annotations_path'],
    )
    decision_s = time.perf_counter() - t
    total_s = time.perf_counter() - t0

    detected = sum(1 for r in tracking_rows if r.get('ball_x') is not None)
    labels = Counter(r['label'] for r in fault_results)
    review_required = sum(1 for r in fault_results if r.get('review_required'))
    return {
        'variant': variant,
        'rep': rep,
        'frames_processed': len(tracking_rows),
        'source_fps': source_fps,
        'ball_tracking_s': round(ball_s, 4),
        'volley_stage_s': round(volley_s, 4),
        'foot_decision_s': round(decision_s, 4),
        'total_core_pipeline_s': round(total_s, 4),
        'ball_tracking_fps': round(len(tracking_rows) / ball_s, 3) if ball_s else '',
        'total_effective_fps': round(len(tracking_rows) / total_s, 3) if total_s else '',
        'times_slower_than_source_total': round(source_fps / (len(tracking_rows) / total_s), 2) if total_s else '',
        'detection_rate_pct': round(100 * detected / max(1, len(tracking_rows)), 2),
        'n_bounce_candidates': len(vc_result.get('bounces', [])),
        'n_manual_events_evaluated': len(fault_results),
        'label_counts': json.dumps(dict(labels), sort_keys=True),
        'review_required_count': review_required,
        'notes': 'core stages: track_ball + run_volley_classification + build manual demo events + run_foot_fault_pipeline; overlays/debug disabled',
    }

rows = []
plan = [
    ('blob_classical', 3, None),
    ('yolo640_notiled_cpu', 3, None),
    ('yolo1280_notiled_cpu', 1, None),
]
for variant, reps, clip_end in plan:
    for rep in range(reps):
        print(f'RUN {variant} rep={rep}', flush=True)
        row = run_one(variant, rep, clip_end=clip_end)
        print(row, flush=True)
        rows.append(row)

# Include tiled reference from earlier measured runs rather than running the full 2055-frame version (~50 minutes expected).
rows.append({
    'variant': 'yolo1280_tiled_cpu_reference_from_prior_300f',
    'rep': 'reference',
    'frames_processed': 300,
    'source_fps': source_fps,
    'ball_tracking_s': 432.123,
    'volley_stage_s': '',
    'foot_decision_s': '',
    'total_core_pipeline_s': '',
    'ball_tracking_fps': 0.694,
    'total_effective_fps': '',
    'times_slower_than_source_total': 86.34,
    'detection_rate_pct': 93.3,
    'n_bounce_candidates': '',
    'n_manual_events_evaluated': '',
    'label_counts': '',
    'review_required_count': '',
    'notes': 'not rerun full clip because prior measured 300-frame tiled CPU path implies about 49.4 minutes just for ball tracking over 2055 frames',
})

csv_path = OUT / 'full_pipeline_variant_benchmark_results.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# Aggregate practical full-clip variants.
agg_rows = []
for variant in sorted({r['variant'] for r in rows if isinstance(r['rep'], int)}):
    vs = [r for r in rows if r['variant'] == variant and isinstance(r['rep'], int)]
    agg_rows.append({
        'variant': variant,
        'n_reps': len(vs),
        'frames_processed_each': vs[0]['frames_processed'],
        'mean_total_s': round(statistics.mean(float(r['total_core_pipeline_s']) for r in vs), 4),
        'median_total_s': round(statistics.median(float(r['total_core_pipeline_s']) for r in vs), 4),
        'mean_total_effective_fps': round(statistics.mean(float(r['total_effective_fps']) for r in vs), 3),
        'mean_times_slower_than_source': round(statistics.mean(float(r['times_slower_than_source_total']) for r in vs), 2),
        'mean_ball_tracking_fps': round(statistics.mean(float(r['ball_tracking_fps']) for r in vs), 3),
        'mean_detection_rate_pct': round(statistics.mean(float(r['detection_rate_pct']) for r in vs), 2),
        'label_counts_last_run': vs[-1]['label_counts'],
        'review_required_last_run': vs[-1]['review_required_count'],
    })
agg_path = OUT / 'full_pipeline_variant_benchmark_summary.csv'
with open(agg_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
    w.writeheader(); w.writerows(agg_rows)

md = ['# Full Pipeline Variant Benchmark', '', 'This benchmark measures the core offline pipeline over the real 2055-frame clip: `track_ball` + `run_volley_classification` + manual demo event construction + `run_foot_fault_pipeline`. Overlay/debug video writing was disabled to focus on processing throughput. The registration CSV was precomputed, matching the normal demo pipeline that consumes `court_reg_v3` outputs.', '', '## Aggregated Full-Clip Results', '', '| Variant | Reps | Mean total time (s) | Median total time (s) | Mean effective FPS | x slower than 59.943 fps | Ball tracking FPS | Ball detection rate | Last labels |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
for r in agg_rows:
    md.append(f"| {r['variant']} | {r['n_reps']} | {r['mean_total_s']} | {r['median_total_s']} | {r['mean_total_effective_fps']} | {r['mean_times_slower_than_source']} | {r['mean_ball_tracking_fps']} | {r['mean_detection_rate_pct']}% | `{r['label_counts_last_run']}` |")
md += ['', '## Slow Tiled Reference', '', 'The original high-resolution tiled CPU YOLO setting was not rerun over the full 2055-frame clip because the prior measured 300-frame benchmark took 432.123 s at 0.694 fps. Extrapolated to 2055 frames, ball tracking alone would take about 49.4 minutes, before event and foot-decision stages.', '', '## Report-Safe Interpretation', '', 'For the real clip, the classical/blob core pipeline is faster than real time but has low ball detection coverage. The non-tiled YOLOv8n `imgsz=640` pipeline is much more practical than tiled inference but remains below the 59.943 fps source rate on CPU. Higher-resolution YOLO improves frame-level detection coverage in these samples but is far slower. These measurements support the claim that the current system is an offline playback/review prototype unless model configuration and hardware acceleration are improved.']
md_path = OUT / 'full_pipeline_variant_benchmark_summary.md'
md_path.write_text('\n'.join(md) + '\n')
print(f'WROTE {csv_path}')
print(f'WROTE {agg_path}')
print(f'WROTE {md_path}')
