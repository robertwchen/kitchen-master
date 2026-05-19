"""
Phase 1 v2 — Court Registration with ORB + RANSAC Homography.

For each frame, estimates the homography from the reference frame to the
current frame (ORB features, BFMatcher + Lowe ratio, RANSAC). Reference
kitchen line geometry is warped through the homography instead of running
Hough per frame. Optionally refines warped line position with a local
perpendicular Sobel search.

Outputs
-------
per_frame_transforms.csv   — frame_index, timestamp, H matrix (9 elements),
                             n_matches, n_inliers, status, near/far line
                             endpoints after warp + refinement
debug_frames/              — annotated PNGs at selected frame indices
overlay.mp4                — annotated video (half-res, 10 fps)
summary_report.json        — per-line stats, fallback counts, method info
comparison_report.json     — v1 (static) vs v2 (registered) edge strength
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Allow running as script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.court_registration import LineModel
from src.stabilizer import FrameStabilizer, refine_line_roi

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── color palette (BGR) ─────────────────────────────────────────────────────
COLOR_NEAR = (0, 220, 50)
COLOR_FAR = (0, 180, 255)
COLOR_FALLBACK = (0, 0, 200)   # red tint when using fallback H
COLOR_TEXT = (255, 255, 255)
COLOR_LEGAL_FILL = np.array([0, 255, 0], dtype=np.float32)
ALPHA_FILL = 0.10


# ── annotation helpers ───────────────────────────────────────────────────────

def _load_annotations(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _ref_endpoints(ann: dict) -> dict:
    """Return {near: (p1,p2), far: (p1,p2), legal_ref: pt, ref_frame: int}."""
    frames = ann["annotated_frames"]
    ref_idx = ann.get("reference_frame_index", frames[0]["frame_index"])
    frame_data = next(
        (f for f in frames if f["frame_index"] == ref_idx), frames[0]
    )
    return {
        "near": (
            tuple(frame_data["near_kitchen_line"]["p1"]),
            tuple(frame_data["near_kitchen_line"]["p2"]),
        ),
        "far": (
            tuple(frame_data["far_kitchen_line"]["p1"]),
            tuple(frame_data["far_kitchen_line"]["p2"]),
        ),
        "legal_ref": tuple(frame_data.get("legal_side_reference_point", [300, 750])),
        "ref_frame_index": ref_idx,
    }


# ── drawing helpers ──────────────────────────────────────────────────────────

def _draw_lines(
    frame: np.ndarray,
    near_pts: tuple,
    far_pts: tuple,
    legal_sign: int,
    fallback: bool = False,
) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    near_color = COLOR_FALLBACK if fallback else COLOR_NEAR
    far_color = COLOR_FALLBACK if fallback else COLOR_FAR

    near_p1, near_p2 = near_pts
    far_p1, far_p2 = far_pts

    # Legal zone fill (vectorized signed distance from near line)
    near_line = LineModel(near_p1, near_p2)
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dist = near_line.a * xx + near_line.b * yy + near_line.c
    mask = legal_sign * dist > 0
    out[mask] = (
        out[mask].astype(np.float32) * (1 - ALPHA_FILL)
        + COLOR_LEGAL_FILL * ALPHA_FILL
    ).astype(np.uint8)

    # Kitchen lines
    cv2.line(out, _ipt(near_p1), _ipt(near_p2), near_color, 2)
    cv2.line(out, _ipt(far_p1), _ipt(far_p2), far_color, 2)

    # Labels
    for label, p1, p2, color in [
        ("near NVZ", near_p1, near_p2, near_color),
        ("far NVZ", far_p1, far_p2, far_color),
    ]:
        mid_x = int((p1[0] + p2[0]) / 2)
        mid_y = int((p1[1] + p2[1]) / 2)
        cv2.putText(out, label, (mid_x - 40, mid_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return out


def _draw_info(
    frame: np.ndarray,
    frame_index: int,
    ts: float,
    n_matches: int,
    n_inliers: int,
    status: str,
    fallback: bool,
) -> np.ndarray:
    out = frame.copy()
    extra = f"matches={n_matches} inliers={n_inliers} [{status}]"
    if fallback:
        extra += " FALLBACK"
    text = f"f={frame_index}  t={ts:.2f}s  {extra}"
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    return out


def _ipt(pt: tuple) -> tuple[int, int]:
    return int(round(pt[0])), int(round(pt[1]))


def _scale_pt(pt: tuple, scale: float) -> tuple[float, float]:
    return pt[0] * scale, pt[1] * scale


# ── edge-strength measurement ─────────────────────────────────────────────────

def _edge_strength_at_line(gray: np.ndarray, p1: tuple, p2: tuple) -> float:
    H, W = gray.shape
    sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.line(mask, _ipt(p1), _ipt(p2), 255, 7)
    vals = sobel[mask > 0]
    return float(vals.mean()) if len(vals) > 0 else 0.0


# ── legal-side sign ───────────────────────────────────────────────────────────

def _legal_sign(near_p1: tuple, near_p2: tuple, ref_pt: tuple) -> int:
    line = LineModel(near_p1, near_p2)
    d = line.signed_distance(ref_pt)
    return 1 if d >= 0 else -1


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run_name"]
    video_path = Path(cfg["video"]["path"])
    ann_path = Path(cfg["annotations"]["path"])
    results_dir = Path(cfg["output"]["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = results_dir / "debug_frames"
    debug_dir.mkdir(exist_ok=True)

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    if not ann_path.exists():
        logger.error(f"Annotations not found: {ann_path}")
        sys.exit(1)

    ann = _load_annotations(ann_path)
    ref = _ref_endpoints(ann)
    near_ref_p1, near_ref_p2 = ref["near"]
    far_ref_p1, far_ref_p2 = ref["far"]
    legal_ref_pt = ref["legal_ref"]
    ref_frame_idx = ref["ref_frame_index"]

    legal_sign = _legal_sign(near_ref_p1, near_ref_p2, legal_ref_pt)
    logger.info(f"Legal-side sign: {legal_sign:+d}")

    # ── load reference frame ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / src_fps

    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ret, ref_frame = cap.read()
    if not ret:
        logger.error(f"Cannot read reference frame {ref_frame_idx}")
        sys.exit(1)

    # ── initialise stabilizer ─────────────────────────────────────────────────
    stab_cfg = cfg.get("stabilizer", {})
    stabilizer = FrameStabilizer(
        n_features=stab_cfg.get("n_features", 3000),
        ratio_test=stab_cfg.get("ratio_test", 0.75),
        min_matches=stab_cfg.get("min_matches", 15),
        ransac_threshold_px=stab_cfg.get("ransac_threshold_px", 4.0),
        top_mask_frac=stab_cfg.get("top_mask_frac", 0.25),
        transform_type=stab_cfg.get("transform_type", "homography"),
    )
    stabilizer.set_reference(ref_frame)

    refine_cfg = cfg.get("refinement", {})
    do_refine = refine_cfg.get("enabled", True)
    refine_search_px = refine_cfg.get("search_px", 20)
    refine_n_pts = refine_cfg.get("n_sample_points", 30)

    debug_indices = set(cfg["output"].get("debug_frame_indices", []))

    # ── first pass: process all frames ───────────────────────────────────────
    logger.info(f"Processing {total_frames} frames …")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows = []
    prev_H = np.eye(3, dtype=np.float64)
    n_fallback = 0
    n_ok = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / src_fps
        H_mat, info = stabilizer.estimate_transform(frame)
        fallback = H_mat is None

        if fallback:
            H_mat = prev_H
            n_fallback += 1
        else:
            prev_H = H_mat
            n_ok += 1

        # Warp reference endpoints
        near_p1_w, near_p2_w = FrameStabilizer.warp_line(near_ref_p1, near_ref_p2, H_mat)
        far_p1_w, far_p2_w = FrameStabilizer.warp_line(far_ref_p1, far_ref_p2, H_mat)

        near_refine = 0
        far_refine = 0
        if do_refine and not fallback:
            near_refine = refine_line_roi(frame, near_p1_w, near_p2_w,
                                          refine_search_px, refine_n_pts)
            far_refine = refine_line_roi(frame, far_p1_w, far_p2_w,
                                         refine_search_px, refine_n_pts)
            line = LineModel(near_p1_w, near_p2_w)
            near_p1_w = (near_p1_w[0] + near_refine * line.a,
                         near_p1_w[1] + near_refine * line.b)
            near_p2_w = (near_p2_w[0] + near_refine * line.a,
                         near_p2_w[1] + near_refine * line.b)
            line_f = LineModel(far_p1_w, far_p2_w)
            far_p1_w = (far_p1_w[0] + far_refine * line_f.a,
                        far_p1_w[1] + far_refine * line_f.b)
            far_p2_w = (far_p2_w[0] + far_refine * line_f.a,
                        far_p2_w[1] + far_refine * line_f.b)

        H_flat = H_mat.flatten().tolist()
        row = {
            "frame_index": frame_idx,
            "timestamp_s": round(ts, 4),
            "H00": H_flat[0], "H01": H_flat[1], "H02": H_flat[2],
            "H10": H_flat[3], "H11": H_flat[4], "H12": H_flat[5],
            "H20": H_flat[6], "H21": H_flat[7], "H22": H_flat[8],
            "n_matches": info["n_matches"],
            "n_inliers": info["n_inliers"],
            "status": info["status"],
            "fallback": int(fallback),
            "near_p1_x": round(near_p1_w[0], 2),
            "near_p1_y": round(near_p1_w[1], 2),
            "near_p2_x": round(near_p2_w[0], 2),
            "near_p2_y": round(near_p2_w[1], 2),
            "far_p1_x": round(far_p1_w[0], 2),
            "far_p1_y": round(far_p1_w[1], 2),
            "far_p2_x": round(far_p2_w[0], 2),
            "far_p2_y": round(far_p2_w[1], 2),
            "near_refine_offset_px": near_refine,
            "far_refine_offset_px": far_refine,
        }
        rows.append(row)

        # Debug frame
        if frame_idx in debug_indices:
            annotated = _draw_lines(
                frame, (near_p1_w, near_p2_w), (far_p1_w, far_p2_w), legal_sign, fallback
            )
            annotated = _draw_info(annotated, frame_idx, ts,
                                    info["n_matches"], info["n_inliers"],
                                    info["status"], fallback)
            dbg_path = debug_dir / f"frame_{frame_idx:05d}.png"
            cv2.imwrite(str(dbg_path), annotated)
            logger.info(f"  Debug frame: {dbg_path}")

        if frame_idx % 500 == 0:
            logger.info(f"  … frame {frame_idx}/{total_frames}  "
                        f"ok={n_ok}  fallback={n_fallback}")

        frame_idx += 1

    cap.release()
    logger.info(f"First pass done: {n_ok} registered, {n_fallback} fallbacks")

    # ── write CSV ─────────────────────────────────────────────────────────────
    csv_path = results_dir / "per_frame_transforms.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"CSV saved: {csv_path}  ({len(rows)} rows)")

    # ── overlay video ─────────────────────────────────────────────────────────
    if cfg["output"].get("save_overlay_video", True):
        overlay_path = results_dir / "overlay.mp4"
        out_fps = float(cfg["output"].get("overlay_video_fps", 10.0))
        scale = float(cfg["output"].get("overlay_video_scale", 0.5))
        frame_step = int(cfg["output"].get("overlay_frame_step", 6))
        out_W = int(src_W * scale)
        out_H = int(src_H * scale)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, out_fps, (out_W, out_H))

        cap = cv2.VideoCapture(str(video_path))
        written = 0
        frame_idx2 = 0
        logger.info("Writing overlay video …")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx2 % frame_step == 0 and frame_idx2 < len(rows):
                r = rows[frame_idx2]
                np1w = (r["near_p1_x"] * scale, r["near_p1_y"] * scale)
                np2w = (r["near_p2_x"] * scale, r["near_p2_y"] * scale)
                fp1w = (r["far_p1_x"] * scale, r["far_p1_y"] * scale)
                fp2w = (r["far_p2_x"] * scale, r["far_p2_y"] * scale)
                small = cv2.resize(frame, (out_W, out_H))
                small = _draw_lines(small, (np1w, np2w), (fp1w, fp2w),
                                    legal_sign, bool(r["fallback"]))
                small = _draw_info(small, frame_idx2, r["timestamp_s"],
                                   r["n_matches"], r["n_inliers"],
                                   r["status"], bool(r["fallback"]))
                writer.write(small)
                written += 1
            frame_idx2 += 1
        cap.release()
        writer.release()
        logger.info(f"Overlay video: {overlay_path}  ({written} frames)")

    # ── compute stability stats ───────────────────────────────────────────────
    def _stability(col_p1x, col_p1y, col_p2x, col_p2y):
        cap2 = cv2.VideoCapture(str(video_path))
        n_sample = min(cfg["comparison"].get("n_sample_frames", 50), len(rows))
        sample_idx = np.linspace(0, len(rows) - 1, n_sample, dtype=int)
        strengths = []
        for si in sample_idx:
            r = rows[si]
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame_index"]))
            ret2, frm = cap2.read()
            if not ret2:
                continue
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            p1 = (r[col_p1x], r[col_p1y])
            p2 = (r[col_p2x], r[col_p2y])
            strengths.append(_edge_strength_at_line(gray, p1, p2))
        cap2.release()
        if not strengths:
            return {}
        arr = np.array(strengths)
        mean_s = float(arr.mean())
        std_s = float(arr.std())
        return {
            "mean_edge_strength": round(mean_s, 2),
            "std_edge_strength": round(std_s, 2),
            "cv": round(std_s / (mean_s + 1e-6), 4),
            "n_frames_sampled": len(strengths),
            "assessment": "stable" if std_s / (mean_s + 1e-6) < 0.15 else "check",
        }

    logger.info("Computing stability stats …")
    near_stability = _stability("near_p1_x", "near_p1_y", "near_p2_x", "near_p2_y")
    far_stability = _stability("far_p1_x", "far_p1_y", "far_p2_x", "far_p2_y")

    overall = (
        "stable"
        if near_stability.get("assessment") == "stable"
        and far_stability.get("assessment") == "stable"
        else "check"
    )

    # ── summary report ────────────────────────────────────────────────────────
    summary = {
        "run_name": run_name,
        "video": video_path.name,
        "resolution": f"{src_W}x{src_H}",
        "fps": round(src_fps, 3),
        "total_frames": total_frames,
        "duration_s": round(duration_s, 2),
        "method": "ORB+RANSAC homography",
        "reference_frame_index": ref_frame_idx,
        "annotation_source": str(ann_path),
        "stabilizer": stab_cfg,
        "registration": {
            "n_ok": n_ok,
            "n_fallback": n_fallback,
            "fallback_rate": round(n_fallback / max(1, total_frames), 4),
        },
        "stability": {
            "near": near_stability,
            "far": far_stability,
        },
        "overall_assessment": overall,
        "legal_side_sign": legal_sign,
        "outputs": {
            "per_frame_transforms_csv": str(csv_path),
            "debug_frames": str(debug_dir),
            "overlay_video": str(results_dir / "overlay.mp4"),
        },
    }
    summary_path = results_dir / "summary_report.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary_path}")

    # ── comparison report ─────────────────────────────────────────────────────
    cmp_cfg = cfg.get("comparison", {})
    if cmp_cfg.get("enabled", True):
        v1_summary_path = Path(cmp_cfg.get("v1_summary", ""))
        v1_near_mean = v1_near_std = v1_far_mean = v1_far_std = None
        if v1_summary_path.exists():
            with open(v1_summary_path) as f:
                v1 = json.load(f)
            v1_near_mean = v1.get("stability", {}).get("near", {}).get("mean_edge_strength")
            v1_near_std = v1.get("stability", {}).get("near", {}).get("std_edge_strength")
            v1_far_mean = v1.get("stability", {}).get("far", {}).get("mean_edge_strength")
            v1_far_std = v1.get("stability", {}).get("far", {}).get("std_edge_strength")

        comparison = {
            "run_name_v1": "court_reg_v1",
            "run_name_v2": run_name,
            "near": {
                "v1_mean_edge_strength": v1_near_mean,
                "v1_std_edge_strength": v1_near_std,
                "v2_mean_edge_strength": near_stability.get("mean_edge_strength"),
                "v2_std_edge_strength": near_stability.get("std_edge_strength"),
                "v2_cv": near_stability.get("cv"),
            },
            "far": {
                "v1_mean_edge_strength": v1_far_mean,
                "v1_std_edge_strength": v1_far_std,
                "v2_mean_edge_strength": far_stability.get("mean_edge_strength"),
                "v2_std_edge_strength": far_stability.get("std_edge_strength"),
                "v2_cv": far_stability.get("cv"),
            },
            "registration_quality": {
                "n_ok": n_ok,
                "n_fallback": n_fallback,
                "fallback_rate": round(n_fallback / max(1, total_frames), 4),
            },
        }
        cmp_path = results_dir / "comparison_report.json"
        with open(cmp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison report: {cmp_path}")

        # Print summary to console
        print("\n── v1 vs v2 edge strength comparison ──────────────────────────")
        for side in ("near", "far"):
            c = comparison[side]
            print(f"  {side}:  v1 mean={c['v1_mean_edge_strength']}  "
                  f"v2 mean={c['v2_mean_edge_strength']}  "
                  f"v2 CV={c['v2_cv']}")
        q = comparison["registration_quality"]
        print(f"\n  Registration: {q['n_ok']} ok  "
              f"{q['n_fallback']} fallback  "
              f"({q['fallback_rate']*100:.1f}% fallback rate)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Court registration v2 (ORB homography)"
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/court_reg_v2.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    run(Path(args.config))
