
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
