
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.foot_fault_pipeline import run_foot_fault_pipeline

OUT = ROOT / 'tmp' / 'final_report_latency_benchmarks'
RESULTS = OUT / 'time_to_decision_results'
RESULTS.mkdir(parents=True, exist_ok=True)

with open(ROOT / 'FINAL_REPORT_HANDOFF_COMPLETE/misc_analysis/demo_pipeline_active_side.yaml') as f:
    cfg = yaml.safe_load(f)

video_path = ROOT / cfg['video']['path']
reg_csv = ROOT / cfg['registration']['csv_path']
ann_path = ROOT / cfg['registration']['annotations_path']
source_fps = 59.943
frame_ms = 1000.0 / source_fps
manual_frames = [int(x) for x in cfg['foot_fault']['manual_volley_frames']]

# Events mirror the active-side demo review. Supplying active_side avoids measuring a different problem.
event_overrides = {
    929: {'override_active_side': 'right'},
    1537: {'override_active_side': 'right'},
    1948: {'override_active_side': 'left', 'ball_x': 449.4, 'ball_y': 504.91},
}
volley_events = []
for fi in manual_frames:
    event = {
        'frame_index': fi,
        'timestamp_s': fi / source_fps,
        'event_type': 'volley_candidate',
    }
    event.update(event_overrides.get(fi, {}))
    volley_events.append(event)

ff_cfg = dict(cfg['foot_fault'])
ff_cfg['foot_localizer'] = cfg['foot_localizer']

# Benchmark repeated full 3-event decision runs. First run may include cold caches/background model state.
rows = []
run_times = []
for i in range(5):
    out_dir = RESULTS / f'run_{i:02d}'
    start = time.perf_counter()
    results = run_foot_fault_pipeline(
        volley_events=volley_events,
        video_path=video_path,
        output_dir=out_dir,
        cfg=ff_cfg,
        registration_csv=reg_csv,
        manual_line_override_path=None,
        ref_annotations_path=ann_path,
    )
    elapsed = time.perf_counter() - start
    run_times.append(elapsed)
    rows.append({
        'run_index': i,
        'n_events': len(results),
        'wall_time_s': round(elapsed, 4),
        'events_per_s': round(len(results) / elapsed, 4) if elapsed else '',
        'ms_per_event_avg': round(1000 * elapsed / max(1, len(results)), 2),
        'labels': json.dumps({r['frame_index']: r['label'] for r in results}, sort_keys=True),
    })

# Per-event isolated benchmark to estimate decision compute per event after imports/caches.
per_event_rows = []
for event in volley_events:
    samples = []
    labels = []
    for j in range(3):
        out_dir = RESULTS / f"event_{event['frame_index']}_{j}"
        start = time.perf_counter()
        res = run_foot_fault_pipeline(
            volley_events=[event],
            video_path=video_path,
            output_dir=out_dir,
            cfg=ff_cfg,
            registration_csv=reg_csv,
            manual_line_override_path=None,
            ref_annotations_path=ann_path,
        )
        elapsed = time.perf_counter() - start
        samples.append(elapsed)
        labels.extend([r['label'] for r in res])
    per_event_rows.append({
        'frame_index': event['frame_index'],
        'timestamp_s': round(event['frame_index'] / source_fps, 4),
        'active_side': event.get('active_side', ''),
        'mean_wall_time_s': round(statistics.mean(samples), 4),
        'median_wall_time_s': round(statistics.median(samples), 4),
        'mean_ms_to_decision_compute': round(1000 * statistics.mean(samples), 2),
        'label_samples': json.dumps(labels),
    })

# Algorithmic delay budget from config windows.
fl = cfg['foot_localizer']
ff = cfg['foot_fault']
vc = cfg['volley_classification']
algorithmic_rows = [
    {
        'stage': 'foot_localization_temporal_smoothing',
        'config': f"temporal_window_radius={fl.get('temporal_window_radius')}",
        'future_frames_needed': int(fl.get('temporal_window_radius', 0)),
        'context_frames_total': 2 * int(fl.get('temporal_window_radius', 0)) + 1,
        'algorithmic_delay_ms': round(int(fl.get('temporal_window_radius', 0)) * frame_ms, 2),
        'meaning': 'needs neighboring frames around event to smooth foot contact point',
    },
    {
        'stage': 'bounce_confirmation',
        'config': f"lookahead_frames={vc.get('lookahead_frames')}",
        'future_frames_needed': int(vc.get('lookahead_frames', 0)),
        'context_frames_total': int(vc.get('lookback_frames', 0)) + int(vc.get('lookahead_frames', 0)) + 1,
        'algorithmic_delay_ms': round(int(vc.get('lookahead_frames', 0)) * frame_ms, 2),
        'meaning': 'needs future ball trajectory to confirm bounce-like reversal',
    },
    {
        'stage': 'active_side_context',
        'config': f"active_side_window_frames={ff.get('active_side_window_frames', 12)}",
        'future_frames_needed': 0,
        'context_frames_total': int(ff.get('active_side_window_frames', 12)),
        'algorithmic_delay_ms': 0.0,
        'meaning': 'uses ball context near event; exact causal delay depends on whether implemented one-sided or centered in deployment',
    },
    {
        'stage': 'hit_classification_lookback',
        'config': f"hit_lookback_frames={vc.get('hit_lookback_frames')}",
        'future_frames_needed': 0,
        'context_frames_total': int(vc.get('hit_lookback_frames', 0)),
        'algorithmic_delay_ms': 0.0,
        'meaning': 'uses past bounce history before a hit; no future delay if hit frame is already known',
    },
]

# Combined estimate: after a hit/event frame is known, foot decision needs the foot smoothing future radius.
combined = {
    'source_fps': source_fps,
    'frame_interval_ms': round(frame_ms, 2),
    'foot_decision_future_frames_after_event_known': int(fl.get('temporal_window_radius', 0)),
    'foot_decision_algorithmic_delay_ms_after_event_known': round(int(fl.get('temporal_window_radius', 0)) * frame_ms, 2),
    'bounce_decision_future_frames_if_event_must_be_inferred': int(vc.get('lookahead_frames', 0)),
    'bounce_decision_algorithmic_delay_ms_if_event_must_be_inferred': round(int(vc.get('lookahead_frames', 0)) * frame_ms, 2),
    'mean_compute_ms_per_event_isolated': round(statistics.mean(float(r['mean_ms_to_decision_compute']) for r in per_event_rows), 2),
    'median_three_event_run_wall_time_s': round(statistics.median(run_times), 4),
    'mean_three_event_ms_per_event': round(1000 * statistics.mean(run_times) / 3, 2),
}

# Write outputs.
with open(OUT / 'time_to_decision_run_benchmark.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(OUT / 'time_to_decision_per_event_benchmark.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(per_event_rows[0].keys()))
    w.writeheader(); w.writerows(per_event_rows)
with open(OUT / 'time_to_decision_algorithmic_delay.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(algorithmic_rows[0].keys()))
    w.writeheader(); w.writerows(algorithmic_rows)
with open(OUT / 'time_to_decision_summary.json', 'w') as f:
    json.dump(combined, f, indent=2)

md = ['# Time-To-Decision Benchmark', '', 'Measured for the three active-side demo events using the existing `run_foot_fault_pipeline` event-decision stage. These timings are offline compute measurements on this machine, not a deployed streaming benchmark.', '', '## Compute Time', '', '| Metric | Value |', '|---|---:|']
md += [
    f"| Median 3-event decision run | {combined['median_three_event_run_wall_time_s']} s |",
    f"| Mean compute per event across 3-event runs | {combined['mean_three_event_ms_per_event']} ms/event |",
    f"| Mean isolated compute per event | {combined['mean_compute_ms_per_event_isolated']} ms/event |",
]
md += ['', '## Per-Event Isolated Compute', '', '| Event frame | Timestamp (s) | Active side | Mean compute ms | Median compute s | Labels observed |', '|---:|---:|---|---:|---:|---|']
for r in per_event_rows:
    md.append(f"| {r['frame_index']} | {r['timestamp_s']} | {r['active_side']} | {r['mean_ms_to_decision_compute']} | {r['median_wall_time_s']} | `{r['label_samples']}` |")
md += ['', '## Algorithmic Delay From Temporal Context', '', '| Stage | Future frames needed | Delay at 59.943 fps | Meaning |', '|---|---:|---:|---|']
for r in algorithmic_rows:
    md.append(f"| {r['stage']} | {r['future_frames_needed']} | {r['algorithmic_delay_ms']} ms | {r['meaning']} |")
md += ['', '## Report-Safe Interpretation', '', f"After a hit/event frame is known, the configured foot decision uses a ±{combined['foot_decision_future_frames_after_event_known']}-frame temporal window, which implies about {combined['foot_decision_algorithmic_delay_ms_after_event_known']} ms of future-frame algorithmic delay at 59.943 fps. The measured event-decision compute stage averaged about {combined['mean_compute_ms_per_event_isolated']} ms per isolated event on this machine. If the event itself must be inferred from ball trajectory, the bounce logic can add up to {combined['bounce_decision_algorithmic_delay_ms_if_event_must_be_inferred']} ms of future-frame delay from `lookahead_frames`. End-to-end live deployment would still need a streaming benchmark that includes ball detection, event inference, foot localization, rendering/output, and p95/p99 latency."]
(OUT / 'time_to_decision_benchmark_summary.md').write_text('\n'.join(md) + '\n')

print(json.dumps(combined, indent=2))
print(f"WROTE {OUT / 'time_to_decision_benchmark_summary.md'}")
