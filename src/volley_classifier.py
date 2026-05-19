"""
Bounce / volley classifier.

Reads per-frame ball tracking data and classifies each hit event as:
  volley          — ball was in the air when struck (no preceding bounce)
  post_bounce_hit — ball bounced before being struck
  uncertain       — cannot determine reliably

Algorithm (interpretable first version)
----------------------------------------
1. Smooth the y-trajectory with a Gaussian kernel.
2. Compute per-frame vertical velocity vy (pixels/frame, positive = downward in image).
3. Scan for bounce candidates: local y-maximum in image coords (ball reached lowest
   point and reversed upward), where vy_before > 0 (falling) and vy_after < 0 (rising).
4. For each candidate check that it occurred near the court surface (within
   court_y_band_px of a reference y supplied from registration).
5. Within a configurable look-back window before each user-identified hit event,
   check whether a confirmed bounce occurred.  If so: post_bounce_hit.  If not: volley.

If no hit events are supplied, the classifier returns all detected bounces as
candidates for manual inspection.

Exports
-------
volley_events/candidates.csv  — all detected bounce candidates + fields for inspection
volley_events/events.csv      — classified hit events (if hit_frames supplied)
volley_events/montage/        — per-candidate 3-panel PNG (before / at / after)
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CANDIDATE_FIELDNAMES = [
    "frame_index",
    "timestamp_s",
    "drop_px",
    "rise_px",
    "vy_before",
    "vy_after",
    "x_consistency_score",
    "local_detection_ratio",
    "max_frame_gap",
    "near_court_surface",
    "confidence",
    "label",
    "ball_x",
    "ball_y",
    "raw_to_smooth_dist_px",
    "mean_window_confidence",
    "dx_before",
    "dx_after",
]

_EVENT_FIELDNAMES = [
    "frame_index",
    "timestamp_s",
    "label",
    "confidence",
    "ball_x",
    "ball_y",
]


# ── smoothing ─────────────────────────────────────────────────────────────────

def _gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return values.copy()
    ksize = max(3, int(sigma * 4) | 1)
    half = ksize // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def _motion_sign(delta: float, min_abs_delta: float) -> int:
    """Return -1 / 0 / +1 depending on whether delta is significant."""
    if abs(delta) < min_abs_delta:
        return 0
    return 1 if delta > 0 else -1


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


def _clear_montage_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for png in output_dir.glob("*.png"):
        png.unlink()


def _dominant_motion_sign(deltas: np.ndarray, min_step_px: float) -> tuple[int, float]:
    """Return dominant x direction and flip ratio over significant local steps."""
    significant = deltas[np.abs(deltas) >= min_step_px]
    if len(significant) == 0:
        return 0, 1.0
    signs = np.sign(significant)
    pos = int(np.sum(signs > 0))
    neg = int(np.sum(signs < 0))
    if pos == neg:
        return 0, 0.5
    dominant = 1 if pos > neg else -1
    flip_ratio = float(np.mean(signs != dominant))
    return dominant, flip_ratio


# ── bounce detection ──────────────────────────────────────────────────────────

def detect_bounces(
    tracking_rows: list[dict],
    cfg: dict,
    court_surface_y: Optional[float | tuple[float, float]] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Detect bounce candidates from per-frame ball tracking rows.

    Parameters
    ----------
    tracking_rows   : output of ball_tracker.track_ball()
    cfg             : volley_classifier config section
    court_surface_y : either a single reference y-pixel or a (far_y, near_y)
                      tuple describing the playable court depth band.
                      If None the proximity gate is skipped.

    Returns `(plausible_candidates, uncertain_candidates)`.
    """
    smooth_sigma = float(cfg.get("smooth_sigma", 2.0))
    min_drop_px = float(cfg.get("min_drop_px", 8.0))
    min_rise_px = float(cfg.get("min_rise_px", 8.0))
    court_band_px = float(cfg.get("court_band_px", cfg.get("court_y_band_px", 120.0)))
    lookback = int(cfg.get("lookback_frames", 5))
    lookahead = int(cfg.get("lookahead_frames", 5))
    uncertain_margin = float(cfg.get("uncertain_margin_px", 20.0))
    max_gap_frames = int(cfg.get("max_frame_gap_in_window", cfg.get("max_detection_gap_frames", 2)))
    min_detection_ratio = float(cfg.get("min_local_detection_ratio", cfg.get("min_detection_ratio", 0.6)))
    min_window_confidence = float(cfg.get("min_window_confidence", 0.45))
    min_x_travel_px = float(cfg.get("min_x_travel_px", 25.0))
    max_x_direction_flip_ratio = float(cfg.get("max_x_direction_flip_ratio", 0.25))
    suppress_stationary_ground_ball = bool(cfg.get("suppress_stationary_ground_ball", True))
    export_uncertain = bool(cfg.get("export_uncertain_candidates", False))
    x_step_significance_px = float(cfg.get("x_step_significance_px", max(4.0, min_x_travel_px / 4.0)))
    stationary_post_x_px = float(cfg.get("stationary_post_x_px", max(30.0, min_x_travel_px * 1.5)))
    stationary_post_gap_frames = int(cfg.get("stationary_post_gap_frames", max_gap_frames))
    max_raw_to_smooth_dist_px = float(cfg.get("max_raw_to_smooth_dist_px", 45.0))

    # Extract detected frames only. Prefer raw Stage 2 detections when available;
    # fall back to the smoothed/public ball_x/ball_y fields otherwise.
    detected = []
    for row in tracking_rows:
        detect_x = row.get("raw_ball_x")
        detect_y = row.get("raw_ball_y")
        detect_conf = row.get("raw_confidence")
        if detect_x is None or detect_y is None:
            detect_x = row.get("ball_x")
            detect_y = row.get("ball_y")
            detect_conf = row.get("confidence", 0.0)
        if detect_y is None:
            continue
        row_copy = dict(row)
        row_copy["_detect_x"] = float(detect_x) if detect_x is not None else None
        row_copy["_detect_y"] = float(detect_y)
        row_copy["_detect_confidence"] = float(detect_conf or 0.0)
        detected.append(row_copy)
    if len(detected) < lookback + lookahead + 1:
        return [], []

    fidx = np.array([r["frame_index"] for r in detected], dtype=np.float64)
    xs = np.array([r["_detect_x"] for r in detected], dtype=np.float64)
    ys = np.array([r["_detect_y"] for r in detected], dtype=np.float64)
    confs = np.array([r.get("_detect_confidence", 0.0) or 0.0 for r in detected], dtype=np.float64)

    sx = _gaussian_smooth(xs, smooth_sigma)
    sy = _gaussian_smooth(ys, smooth_sigma)
    vy = np.gradient(sy, fidx)  # pixels/frame, positive = downward

    plausible_candidates = []
    uncertain_candidates = []
    for i in range(lookback, len(detected) - lookahead):
        frame_i = int(detected[i]["frame_index"])
        next_fidx = fidx[i:i + lookahead + 1]
        local_fidx = fidx[i - lookback:i + lookahead + 1]
        local_sx = sx[i - lookback:i + lookahead + 1]

        # A real bounce needs a mostly continuous local track. Large gaps inflate
        # drop/rise measurements because the old code was measuring over detected
        # points rather than over a contiguous frame window.
        max_local_gap = int(np.max(np.diff(local_fidx))) if len(local_fidx) >= 2 else 0
        gap_ok = max_local_gap <= max_gap_frames

        window_start = frame_i - lookback
        window_end = frame_i + lookahead
        local_rows = [
            r for r in tracking_rows
            if window_start <= int(r["frame_index"]) <= window_end
        ]
        local_detected = [r for r in local_rows if r.get("ball_y") is not None]
        detection_ratio = len(local_detected) / max(1, len(local_rows))
        density_ok = detection_ratio >= min_detection_ratio

        mean_window_conf = float(np.mean(confs[i - lookback:i + lookahead + 1]))
        conf_ok = mean_window_conf >= min_window_confidence

        # local maximum in y (lowest point of ball arc = bounce)
        if sy[i] < sy[i - 1] or sy[i] < sy[i + 1]:
            continue

        vy_before = float(vy[i - 1])
        vy_after = float(vy[i + 1])

        # must be falling before and rising after
        if vy_before <= 0 or vy_after >= 0:
            continue

        # amplitude of drop and rise
        drop_px = float(sy[i] - sy[i - lookback])
        rise_px = float(sy[i] - sy[i + lookahead])

        if drop_px < min_drop_px or rise_px < min_rise_px:
            continue

        smooth_y_pos = float(sy[i])
        smooth_x_pos = float(sx[i])
        raw_x_pos = float(detected[i]["_detect_x"])
        raw_y_pos = float(detected[i]["_detect_y"])
        raw_to_smooth_dist = float(np.hypot(raw_x_pos - smooth_x_pos, raw_y_pos - smooth_y_pos))

        dx_before = float(sx[i] - sx[i - lookback])
        dx_after = float(sx[i + lookahead] - sx[i])
        x_deltas = np.diff(local_sx)
        dominant_x_sign, x_flip_ratio = _dominant_motion_sign(x_deltas, x_step_significance_px)
        x_consistency_score = 1.0 - x_flip_ratio
        x_dir_before = _motion_sign(dx_before, min_x_travel_px)
        x_dir_after = _motion_sign(dx_after, min_x_travel_px)
        x_direction_consistent = (
            x_flip_ratio <= max_x_direction_flip_ratio and
            not (
                x_dir_before != 0 and
                x_dir_after != 0 and
                x_dir_before != x_dir_after
            )
        )

        # proximity to court surface gate
        near_surface = True
        if court_surface_y is not None:
            if isinstance(court_surface_y, tuple):
                far_y, near_y = sorted(float(v) for v in court_surface_y)
                near_surface = (far_y - court_band_px) <= raw_y_pos <= (near_y + court_band_px)
            else:
                near_surface = abs(raw_y_pos - float(court_surface_y)) <= court_band_px

        post_max_gap = int(np.max(np.diff(next_fidx))) if len(next_fidx) >= 2 else 0
        stationary_ground_ball = (
            suppress_stationary_ground_ball and
            abs(dx_after) < stationary_post_x_px and
            post_max_gap > stationary_post_gap_frames
        )

        if stationary_ground_ball:
            x_direction_consistent = False

        raw_anchor_ok = raw_to_smooth_dist <= max_raw_to_smooth_dist_px

        continuity_score = min(1.0, detection_ratio / max(min_detection_ratio, 1e-6))
        confidence_score = min(1.0, mean_window_conf / max(min_window_confidence, 1e-6))

        # confidence: combine bounce amplitude, x consistency, and local track quality
        conf_raw = min(1.0, (drop_px + rise_px) / 60.0)
        conf_raw *= x_consistency_score
        conf_raw *= continuity_score
        conf_raw *= confidence_score
        if not near_surface:
            conf_raw *= 0.5
        if not x_direction_consistent:
            conf_raw *= 0.5
        if not raw_anchor_ok:
            conf_raw *= 0.5

        if (
            near_surface and
            x_direction_consistent and
            raw_anchor_ok and
            gap_ok and
            density_ok and
            conf_ok and
            conf_raw >= 0.35
        ):
            label = "bounce"
        elif conf_raw >= 0.2:
            label = "uncertain"
        else:
            continue

        if stationary_ground_ball:
            label = "uncertain"

        candidate = {
            "frame_index": frame_i,
            "timestamp_s": float(detected[i]["timestamp_s"]),
            "drop_px": round(drop_px, 2),
            "rise_px": round(rise_px, 2),
            "vy_before": round(vy_before, 3),
            "vy_after": round(vy_after, 3),
            "x_consistency_score": round(x_consistency_score, 3),
            "local_detection_ratio": round(detection_ratio, 3),
            "max_frame_gap": max_local_gap,
            "near_court_surface": near_surface,
            "confidence": round(conf_raw, 3),
            "label": label,
            "ball_x": raw_x_pos,
            "ball_y": raw_y_pos,
            "raw_to_smooth_dist_px": round(raw_to_smooth_dist, 2),
            "mean_window_confidence": round(mean_window_conf, 3),
            "dx_before": round(dx_before, 2),
            "dx_after": round(dx_after, 2),
        }

        if label == "bounce":
            plausible_candidates.append(candidate)
        elif export_uncertain:
            uncertain_candidates.append(candidate)

    return plausible_candidates, uncertain_candidates


# ── hit event classification ──────────────────────────────────────────────────

def classify_hit_events(
    hit_frames: list[int],
    bounces: list[dict],
    cfg: dict,
    tracking_rows: list[dict],
) -> list[dict]:
    """
    For each supplied hit frame, decide volley / post_bounce_hit / uncertain.

    A bounce within [hit_frame - window, hit_frame) counts as a preceding bounce.
    """
    window = int(cfg.get("hit_lookback_frames", 30))
    min_bounce_conf = float(cfg.get("min_bounce_confidence", 0.3))

    bounce_frames = {
        b["frame_index"] for b in bounces
        if b["confidence"] >= min_bounce_conf and b["label"] == "bounce"
    }
    uncertain_frames = {
        b["frame_index"] for b in bounces
        if b["confidence"] >= 0.15 and b["label"] == "uncertain"
    }

    rows_by_frame = {r["frame_index"]: r for r in tracking_rows}

    events = []
    for hf in hit_frames:
        tr = rows_by_frame.get(hf, {})
        has_bounce = any(
            (hf - window) <= bf < hf for bf in bounce_frames
        )
        has_uncertain_bounce = any(
            (hf - window) <= bf < hf for bf in uncertain_frames
        )
        if has_bounce:
            label = "post_bounce_hit"
            conf = 0.8
        elif has_uncertain_bounce:
            label = "uncertain"
            conf = 0.5
        else:
            label = "volley"
            conf = 0.75

        events.append({
            "frame_index": hf,
            "timestamp_s": float(tr.get("timestamp_s", hf)),
            "label": label,
            "confidence": conf,
            "ball_x": tr.get("ball_x"),
            "ball_y": tr.get("ball_y"),
        })
    return events


# ── verification montage ──────────────────────────────────────────────────────

def _read_frame(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frm = cap.read()
    return frm if ret else None


def write_bounce_montages(
    video_path: Path,
    candidates: list[dict],
    output_dir: Path,
    context_frames: int = 5,
    scale: float = 0.4,
) -> None:
    """
    For each bounce candidate, write a 3-panel PNG:
      [frame_before | candidate_frame | frame_after]
    with annotation showing vy_before / vy_after / label.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not candidates:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video for montages: {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_W = max(1, int(src_W * scale))
    out_H = max(1, int(src_H * scale))

    for cand in candidates:
        fi = int(cand["frame_index"])
        before_fi = max(0, fi - context_frames)
        after_fi = fi + context_frames

        before = _read_frame(cap, before_fi)
        center = _read_frame(cap, fi)
        after = _read_frame(cap, after_fi)

        panels = []
        for frm, label_str, fii in [
            (before, f"before f={before_fi}", before_fi),
            (center, f"CANDIDATE f={fi}", fi),
            (after,  f"after f={after_fi}", after_fi),
        ]:
            if frm is None:
                frm = np.zeros((src_H, src_W, 3), dtype=np.uint8)
            p = cv2.resize(frm, (out_W, out_H))
            cv2.putText(p, label_str, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(p, label_str, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if fii == fi:
                # annotate ball position and stats
                bx = cand.get("ball_x")
                by = cand.get("ball_y")
                if bx is not None:
                    cx2 = int(round(float(bx) * scale))
                    cy2 = int(round(float(by) * scale))
                    cv2.circle(p, (cx2, cy2), 10, (0, 255, 255), 2)
                info = (
                    f"vy_before={cand['vy_before']:.1f}  "
                    f"vy_after={cand['vy_after']:.1f}  "
                    f"conf={cand['confidence']:.2f}  [{cand['label']}]"
                )
                cv2.putText(p, info, (6, out_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
                cv2.putText(p, info, (6, out_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            panels.append(p)

        montage = np.hstack(panels)
        out_path = output_dir / f"bounce_{fi:05d}.png"
        cv2.imwrite(str(out_path), montage)

    cap.release()
    logger.info(f"Bounce montages written to {output_dir}  ({len(candidates)} candidates)")


# ── top-level entry point ─────────────────────────────────────────────────────

def run_volley_classification(
    tracking_rows: list[dict],
    video_path: Path,
    output_dir: Path,
    cfg: dict,
    court_surface_y: Optional[float] = None,
    hit_frames: Optional[list[int]] = None,
) -> dict:
    """
    Full bounce/volley classification pass.

    Returns dict with keys:
      bounces   — list of detected bounce candidates
      events    — list of classified hit events (empty if hit_frames not given)
    Also writes CSVs and montage PNGs into output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    bounces, uncertain_bounces = detect_bounces(
        tracking_rows, cfg, court_surface_y=court_surface_y
    )
    logger.info(
        f"Bounce detection: {len(bounces)} plausible candidates found"
        + (
            f" (+{len(uncertain_bounces)} uncertain debug candidates)"
            if uncertain_bounces else ""
        )
    )

    # write bounce CSV (always rewrite so stale candidates do not survive)
    cand_csv = output_dir / "candidates.csv"
    _write_csv_rows(cand_csv, _CANDIDATE_FIELDNAMES, bounces)
    if cfg.get("export_uncertain_candidates", False):
        uncertain_csv = output_dir / "uncertain_candidates.csv"
        _write_csv_rows(uncertain_csv, _CANDIDATE_FIELDNAMES, uncertain_bounces)

    # montages
    montage_dir = output_dir / "montage"
    _clear_montage_dir(montage_dir)
    write_bounce_montages(video_path, bounces, montage_dir)
    if cfg.get("export_uncertain_candidates", False):
        uncertain_montage_dir = output_dir / "uncertain_montage"
        _clear_montage_dir(uncertain_montage_dir)
        write_bounce_montages(video_path, uncertain_bounces, uncertain_montage_dir)
    else:
        uncertain_montage_dir = output_dir / "uncertain_montage"
        _clear_montage_dir(uncertain_montage_dir)

    events: list[dict] = []
    if hit_frames:
        events = classify_hit_events(hit_frames, bounces, cfg, tracking_rows)
        event_csv = output_dir / "events.csv"
        _write_csv_rows(event_csv, _EVENT_FIELDNAMES, events)
        logger.info(f"Hit events classified: {len(events)} events")

    return {"bounces": bounces, "events": events}
