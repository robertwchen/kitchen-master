"""
Yellow pickleball tracker — optimized for night / artificial-light footage.

Key design decisions (derived from pixel analysis of this specific footage):
  - Under artificial floodlights, the ball reads H≈30-40, S≈5-106 (highly
    variable), V≈200-250.  Saturation alone cannot gate the ball.
  - The primary discriminants are:
      1. MOTION  — frame differencing eliminates all static scene elements
                   (court lights, net reflection, bench, fence blobs).
      2. H band  — yellow-green range H:[20,55] excludes blue fence (H≈17),
                   green clothing (H≈60+), white/gray (S≈0).
      3. High V  — V≥195 excludes dark court surface, shadows, and dark
                   moving blobs from players' legs.
  - The ball area ranges from ~50 px² (distant) to ~2500 px² (close).
    Circularity drops to ≈0.35 under motion blur; static scene elements
    have circularity 0.01-0.30, players have large irregular blobs.

Detection modes (set via cfg['detection_mode']):
  diff_and_hsv  (default)  — frame-diff mask ∩ HSV mask.  Best for moving ball.
  shape_only               — frame-diff + area/circularity, NO color gate.
                             Works for any ball color / unknown footage.
  hsv_only                 — pure HSV threshold.  Useful as fallback / debug.

Candidate scoring:
  All candidates are scored by  circularity × (v_at_centroid / 255).
  This ranks the ball first: it has the highest V among round moving blobs.
  Candidates below min_v_at_centroid (default 180) are rejected outright.

Outputs
-------
per_frame CSV  — frame_index, timestamp_s, ball_x, ball_y, confidence
overlay video  — ball marker + colour-coded trailing arc
debug PNGs     — sampled frames showing detection + masks side-by-side
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.ball_detector import UltralyticsBallDetector, render_ultralytics_debug

logger = logging.getLogger(__name__)

# ── calibrated defaults for night / artificial-light footage ─────────────────
# Ball reads H≈30-40 (yellow-green), S≈5-106 (low and inconsistent under
# floodlights), V≈200-250.  S minimum is very low — rely on H + V instead.
_DEFAULT_HSV_LOWER = np.array([20, 5, 195], dtype=np.uint8)
_DEFAULT_HSV_UPPER = np.array([55, 255, 255], dtype=np.uint8)

# Frame-diff threshold: pixel must change by ≥ this many grey levels to count
_DEFAULT_DIFF_THRESH = 20
# Dilate the diff mask this much to fill the ball region (px)
_DEFAULT_DIFF_DILATE = 12


# ── detection helpers ─────────────────────────────────────────────────────────

def _build_top_mask(H: int, W: int, top_frac: float) -> np.ndarray:
    """Mask off the top portion of the frame (background lights, etc.)."""
    mask = np.ones((H, W), dtype=np.uint8) * 255
    cut = int(H * top_frac)
    if cut > 0:
        mask[:cut, :] = 0
    return mask


def _hsv_mask(frame: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lo, hi)


def _diff_mask(
    gray: np.ndarray,
    prev_gray: np.ndarray,
    diff_thresh: int,
    dilate_k: int,
) -> np.ndarray:
    diff = cv2.absdiff(gray, prev_gray)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    if dilate_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        mask = cv2.dilate(mask, k)
    return mask


def _contour_candidates(
    mask: np.ndarray,
    min_area: float,
    max_area: float,
    min_circ: float,
    hsv_frame: Optional[np.ndarray] = None,
    min_v_at_centroid: Optional[int] = None,
) -> list[dict]:
    """
    Extract ball candidates from a binary mask.

    If hsv_frame is supplied (H×W×3 uint8 in HSV), each candidate is scored
    by circularity × (V_at_centroid / 255) and sorted best-first.
    min_v_at_centroid filters out dark moving blobs (shadows, players).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-6:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circ:
            continue
        M = cv2.moments(cnt)
        if M["m00"] < 1e-6:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        v_at_center: Optional[int] = None
        if hsv_frame is not None:
            py = int(np.clip(round(cy), 0, hsv_frame.shape[0] - 1))
            px = int(np.clip(round(cx), 0, hsv_frame.shape[1] - 1))
            v_at_center = int(hsv_frame[py, px, 2])
            if min_v_at_centroid is not None and v_at_center < min_v_at_centroid:
                continue

        score = circularity * (v_at_center / 255.0 if v_at_center is not None else 1.0)

        candidates.append({
            "x": cx,
            "y": cy,
            "radius": float(np.sqrt(area / np.pi)),
            "area": area,
            "circularity": circularity,
            "v_at_center": v_at_center,
            "score": score,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _apply_morphology(
    mask: np.ndarray,
    open_k: int,
    close_k: int,
) -> np.ndarray:
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _detect_diff_and_hsv(
    frame: np.ndarray,
    prev_gray: np.ndarray,
    hsv_lo: np.ndarray,
    hsv_hi: np.ndarray,
    diff_thresh: int,
    diff_dilate: int,
    morph_open_k: int,
    morph_close_k: int,
    top_mask: np.ndarray,
    min_area: float,
    max_area: float,
    min_circ: float,
    min_v_at_centroid: Optional[int],
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Detect candidates using frame-diff ∩ HSV intersection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dm = _diff_mask(gray, prev_gray, diff_thresh, diff_dilate)
    hm = cv2.inRange(hsv, hsv_lo, hsv_hi)
    combined = cv2.bitwise_and(dm, hm)
    combined = cv2.bitwise_and(combined, top_mask)
    combined = _apply_morphology(combined, morph_open_k, morph_close_k)

    return (
        _contour_candidates(combined, min_area, max_area, min_circ, hsv, min_v_at_centroid),
        combined,
        gray,
    )


def _detect_shape_only(
    frame: np.ndarray,
    prev_gray: np.ndarray,
    diff_thresh: int,
    diff_dilate: int,
    morph_open_k: int,
    morph_close_k: int,
    top_mask: np.ndarray,
    min_area: float,
    max_area: float,
    min_circ: float,
    min_v_at_centroid: Optional[int],
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Detect candidates using frame-diff only — NO color gate.

    This mode works for any ball color as long as the ball is moving.
    Players and large objects are rejected by max_area + circularity.
    Optionally gated by V brightness at centroid to reject dark moving blobs.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if min_v_at_centroid is not None else None

    dm = _diff_mask(gray, prev_gray, diff_thresh, diff_dilate)
    dm = cv2.bitwise_and(dm, top_mask)
    dm = _apply_morphology(dm, morph_open_k, morph_close_k)

    return (
        _contour_candidates(dm, min_area, max_area, min_circ, hsv, min_v_at_centroid),
        dm,
        gray,
    )


def _detect_hsv_only(
    frame: np.ndarray,
    hsv_lo: np.ndarray,
    hsv_hi: np.ndarray,
    morph_open_k: int,
    morph_close_k: int,
    top_mask: np.ndarray,
    min_area: float,
    max_area: float,
    min_circ: float,
    min_v_at_centroid: Optional[int],
) -> tuple[list[dict], np.ndarray]:
    """Detect candidates using HSV threshold only (no frame diff)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hm  = cv2.inRange(hsv, hsv_lo, hsv_hi)
    hm  = cv2.bitwise_and(hm, top_mask)
    hm  = _apply_morphology(hm, morph_open_k, morph_close_k)

    return (
        _contour_candidates(hm, min_area, max_area, min_circ, hsv, min_v_at_centroid),
        hm,
    )


def _detect_ultralytics(
    frame: np.ndarray,
    detector: UltralyticsBallDetector,
    top_mask: np.ndarray,
    min_area: float,
    max_area: float,
) -> tuple[list[dict], np.ndarray]:
    """Learned detector-first candidate generation."""
    candidates = []
    for cand in detector.detect(frame):
        py = int(np.clip(round(cand["y"]), 0, top_mask.shape[0] - 1))
        px = int(np.clip(round(cand["x"]), 0, top_mask.shape[1] - 1))
        if top_mask[py, px] == 0:
            continue
        area = float(cand.get("area") or 0.0)
        if area < min_area or area > max_area:
            continue
        candidates.append(cand)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates, render_ultralytics_debug(frame, candidates)


# ── temporal linking ──────────────────────────────────────────────────────────

def _link(
    prev: Optional[dict],
    candidates: list[dict],
    max_jump_px: float,
) -> Optional[dict]:
    """
    Link to the closest candidate within max_jump_px of prev detection.
    If no prev, return the highest-scoring candidate (best circularity × V).
    """
    if not candidates:
        return None
    if prev is None:
        return candidates[0]
    best, best_d = None, float("inf")
    for c in candidates:
        d = float(np.hypot(c["x"] - prev["x"], c["y"] - prev["y"]))
        if d < best_d:
            best_d, best = d, c
    return best if best_d <= max_jump_px else None


# ── Gaussian trail smoother ───────────────────────────────────────────────────

def _smooth_trail(detections: list[dict], sigma: float) -> list[dict]:
    if sigma <= 0.0:
        return detections
    detected_idx = [i for i, d in enumerate(detections) if d.get("ball_x") is not None]
    if len(detected_idx) < 3:
        return detections

    vx = np.array([detections[i]["ball_x"] for i in detected_idx], dtype=np.float64)
    vy = np.array([detections[i]["ball_y"] for i in detected_idx], dtype=np.float64)

    ksize = max(3, int(sigma * 4) | 1)
    half  = ksize // 2
    t     = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()

    svx = np.convolve(vx, kernel, mode="same")
    svy = np.convolve(vy, kernel, mode="same")

    result = [d.copy() for d in detections]
    for k, i in enumerate(detected_idx):
        result[i]["ball_x"] = round(float(svx[k]), 2)
        result[i]["ball_y"] = round(float(svy[k]), 2)
    return result


# ── debug frame rendering ─────────────────────────────────────────────────────

def _make_debug_frame(
    frame: np.ndarray,
    combined_mask: np.ndarray,
    det: dict,
    trail: list,
    frame_index: int,
    timestamp_s: float,
    scale: float,
    n_candidates: int,
) -> np.ndarray:
    """Side-by-side: annotated frame + detection mask."""
    H, W = frame.shape[:2]
    oW, oH = int(W * scale), int(H * scale)

    # Left panel: annotated frame
    left = cv2.resize(frame, (oW, oH))
    for k, pt in enumerate(trail):
        if pt is None:
            continue
        alpha = (k + 1) / max(1, len(trail))
        color = (int(20 * alpha), int(200 * alpha), int(255 * alpha))
        cv2.circle(left, (int(round(pt[0] * scale)), int(round(pt[1] * scale))), 3, color, -1)
    if det.get("ball_x") is not None:
        cx = int(round(det["ball_x"] * scale))
        cy = int(round(det["ball_y"] * scale))
        r  = max(8, int(round(det.get("radius", 8) * scale)))
        cv2.circle(left, (cx, cy), r, (0, 255, 255), 2)
        cv2.circle(left, (cx, cy), 2, (0, 255, 255), -1)

    bx, by = det.get("ball_x"), det.get("ball_y")
    conf = det.get("confidence", 0.0)
    v_ctr = det.get("v_at_center", "?")
    circ  = det.get("circularity") or 0.0
    info  = (f"f={frame_index} t={timestamp_s:.2f}s  "
             f"ball=({bx},{by}) conf={conf:.2f}  "
             f"circ={circ:.2f} V={v_ctr}  ncand={n_candidates}")
    cv2.putText(left, info, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(left, info, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Right panel: detection mask or detector debug view
    if combined_mask.ndim == 2:
        mask_vis = cv2.resize(combined_mask, (oW, oH), interpolation=cv2.INTER_NEAREST)
        mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        mask_bgr[:, :, 1] = mask_vis  # green channel = detections
    else:
        mask_bgr = cv2.resize(combined_mask, (oW, oH))

    return np.hstack([left, mask_bgr])


# ── public API ────────────────────────────────────────────────────────────────

def track_ball(
    video_path: Path,
    output_dir: Path,
    cfg: dict,
    clip_start_frame: int = 0,
    clip_end_frame: Optional[int] = None,
    debug_every_n: int = 60,
    write_overlay: bool = True,
    overlay_fps: float = 10.0,
    overlay_scale: float = 0.5,
) -> list[dict]:
    """
    Run ball tracking on a video clip.

    Returns list of per-frame row dicts.
    Writes:
      output_dir/ball_tracking.csv
      output_dir/ball_overlay.mp4
      output_dir/debug_frames/frame_NNNNN.png  (side-by-side annotated + mask)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_frames"
    debug_dir.mkdir(exist_ok=True)

    # ── parse config ──────────────────────────────────────────────────────────
    hsv_lo = np.array(cfg.get("hsv_lower", _DEFAULT_HSV_LOWER.tolist()), dtype=np.uint8)
    hsv_hi = np.array(cfg.get("hsv_upper", _DEFAULT_HSV_UPPER.tolist()), dtype=np.uint8)
    min_area           = float(cfg.get("min_area",           20.0))
    max_area           = float(cfg.get("max_area",           2500.0))
    min_circ           = float(cfg.get("min_circularity",    0.30))
    morph_open         = int(cfg.get("morph_open_k",         3))
    morph_close        = int(cfg.get("morph_close_k",        5))
    max_jump           = float(cfg.get("max_jump_px",        120.0))
    smooth_sigma       = float(cfg.get("smooth_sigma",       1.5))
    trail_len          = int(cfg.get("trail_length",         12))
    top_frac           = float(cfg.get("top_exclude_frac",   0.20))
    detection_mode     = str(cfg.get("detection_mode",       "diff_and_hsv"))
    tracking_backend   = str(cfg.get("tracking_backend",     "blob")).lower()
    diff_thresh        = int(cfg.get("diff_threshold",       _DEFAULT_DIFF_THRESH))
    diff_dilate        = int(cfg.get("diff_dilate_k",        _DEFAULT_DIFF_DILATE))
    min_v_at_centroid  = cfg.get("min_v_at_centroid")
    if min_v_at_centroid is not None:
        min_v_at_centroid = int(min_v_at_centroid)
    detector = None
    if tracking_backend == "ultralytics":
        detector = UltralyticsBallDetector.from_config(cfg)
    elif tracking_backend not in {"blob", "classical"}:
        raise ValueError(f"Unsupported tracking_backend: {tracking_backend}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frame = clip_end_frame if clip_end_frame is not None else total

    top_mask = _build_top_mask(src_H, src_W, top_frac)
    out_W    = int(src_W * overlay_scale)
    out_H    = int(src_H * overlay_scale)

    writer = None
    overlay_path = output_dir / "ball_overlay.mp4"
    if write_overlay:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, overlay_fps, (out_W, out_H))

    rows: list[dict] = []
    prev_det:  Optional[dict]  = None
    trail:     list[Optional[tuple[float, float]]] = []
    prev_gray: Optional[np.ndarray] = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)
    frame_idx = clip_start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / src_fps

        # ── detect ────────────────────────────────────────────────────────────
        need_diff = detection_mode in ("diff_and_hsv", "shape_only")

        if tracking_backend == "ultralytics":
            candidates, combined_mask = _detect_ultralytics(
                frame, detector, top_mask, min_area, max_area
            )
        elif need_diff:
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                # Diff-based modes need a previous frame. Do not seed the linker
                # from an HSV-only detection on frame 0; static bright blobs can
                # otherwise hijack the track before any motion signal exists.
                candidates = []
                combined_mask = np.zeros_like(cur_gray)
            elif detection_mode == "diff_and_hsv":
                candidates, combined_mask, _ = _detect_diff_and_hsv(
                    frame, prev_gray, hsv_lo, hsv_hi,
                    diff_thresh, diff_dilate, morph_open, morph_close,
                    top_mask, min_area, max_area, min_circ, min_v_at_centroid,
                )
            else:  # shape_only
                candidates, combined_mask, _ = _detect_shape_only(
                    frame, prev_gray,
                    diff_thresh, diff_dilate, morph_open, morph_close,
                    top_mask, min_area, max_area, min_circ, min_v_at_centroid,
                )
            prev_gray = cur_gray
        else:
            candidates, combined_mask = _detect_hsv_only(
                frame, hsv_lo, hsv_hi, morph_open, morph_close,
                top_mask, min_area, max_area, min_circ, min_v_at_centroid,
            )

        det = _link(prev_det, candidates, max_jump)

        # confidence = score of best candidate (circ × V_norm), or circularity alone
        if det is not None:
            conf = round(float(det.get("detector_confidence", det.get("score") or det.get("circularity", 0.0))), 3)
        else:
            conf = 0.0

        bbox = det.get("bbox") if det else None

        row: dict = {
            "frame_index":  frame_idx,
            "timestamp_s":  round(ts, 4),
            "ball_x":       round(det["x"], 2)           if det else None,
            "ball_y":       round(det["y"], 2)           if det else None,
            "raw_ball_x":   round(det["x"], 2)           if det else None,
            "raw_ball_y":   round(det["y"], 2)           if det else None,
            "radius_px":    round(det["radius"], 2)      if det else None,
            "circularity":  round(det["circularity"], 3) if det and det.get("circularity") is not None else None,
            "v_at_center":  det.get("v_at_center")       if det else None,
            "raw_confidence": conf,
            "bbox_x0":      round(bbox[0], 2) if bbox else None,
            "bbox_y0":      round(bbox[1], 2) if bbox else None,
            "bbox_x1":      round(bbox[2], 2) if bbox else None,
            "bbox_y1":      round(bbox[3], 2) if bbox else None,
            "tracking_backend": tracking_backend,
            "confidence":   conf,
            "n_candidates": len(candidates),
        }
        rows.append(row)
        prev_det = det

        trail.append((det["x"], det["y"]) if det else None)
        if len(trail) > trail_len:
            trail.pop(0)

        # ── output ────────────────────────────────────────────────────────────
        is_debug = (frame_idx - clip_start_frame) % debug_every_n == 0
        if writer is not None or is_debug:
            debug_img = _make_debug_frame(
                frame, combined_mask, row, trail,
                frame_idx, ts, overlay_scale, len(candidates),
            )
            if is_debug:
                dbg_path = debug_dir / f"frame_{frame_idx:05d}.png"
                cv2.imwrite(str(dbg_path), debug_img)

            if writer is not None:
                left_panel = debug_img[:, :out_W, :]
                writer.write(left_panel)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    rows = _smooth_trail(rows, sigma=smooth_sigma)

    csv_path = output_dir / "ball_tracking.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    n_detected = sum(1 for r in rows if r["ball_x"] is not None)
    logger.info(
        f"Ball tracking done: {n_detected}/{len(rows)} frames "
        f"({n_detected / max(1, len(rows)) * 100:.1f}%)  CSV={csv_path}"
    )
    return rows
