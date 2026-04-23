"""
Phase 1 v3 — Court Registration from Anchor Points + ORB Homography.

Strategy
--------
1. User annotates 6–11 anchor points on a clean reference frame:
   near corners, far corners, net ends, optional kitchen-line intersections.
2. CourtGeometryModel derives the full court structure (net, kitchen lines,
   legal zone polygons) from those anchors.
3. ORB + BFMatcher + RANSAC homography estimates the per-frame transform
   relative to the reference frame.
4. The reference CourtGeometryModel is warped through the homography to give
   per-frame positions for every structural element.
5. Optional local Sobel refinement adjusts each kitchen line ±search_px
   perpendicular to the predicted position.
6. Outputs: per-frame CSV, debug PNGs, overlay video, validation report.

Validation report
-----------------
Measures edge strength at projected kitchen line positions across sampled
frames, and records the distribution of homography translation magnitudes to
confirm camera motion is within expected bounds.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.court_model import CourtGeometryModel
from src.court_registration import LineModel
from src.stabilizer import FrameStabilizer, refine_line_roi
from src.viz import draw_court_model, draw_frame_info

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_annotations(path: Path) -> tuple[dict, int]:
    """Return (anchor_dict, reference_frame_index)."""
    with open(path) as f:
        ann = json.load(f)
    ref_idx = ann.get("reference_frame_index", 0)
    frames = ann["annotated_frames"]
    frame_data = next(
        (f for f in frames if f["frame_index"] == ref_idx), frames[0]
    )
    anchors = frame_data["anchors"]
    return anchors, ref_idx


def _ipt(pt) -> tuple[int, int]:
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def _edge_strength(gray: np.ndarray, p1: tuple, p2: tuple) -> float:
    H, W = gray.shape
    sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.line(mask, _ipt(p1), _ipt(p2), 255, 7)
    vals = sobel[mask > 0]
    return float(vals.mean()) if len(vals) else 0.0


def _apply_refinement(
    frame: np.ndarray,
    line: LineModel,
    search_px: int,
    n_pts: int,
    W: int,
    H: int,
) -> tuple[tuple, tuple, int]:
    """Return (new_p1, new_p2, offset_px) after perpendicular refinement."""
    pt1, pt2 = line.endpoints_in_frame(W, H)
    offset = refine_line_roi(frame, pt1, pt2, search_px, n_pts)
    new_p1 = (pt1[0] + offset * line.a, pt1[1] + offset * line.b)
    new_p2 = (pt2[0] + offset * line.a, pt2[1] + offset * line.b)
    return new_p1, new_p2, offset


def _draw_info_v3(
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
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def run(config_path: Path) -> None:
    cfg = _load_config(config_path)
    run_name = cfg["run_name"]
    video_path = Path(cfg["video"]["path"])
    ann_path = Path(cfg["annotations"]["path"])
    results_dir = Path(cfg["output"]["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = results_dir / "debug_frames"
    debug_dir.mkdir(exist_ok=True)

    for p, label in [(video_path, "video"), (ann_path, "annotations")]:
        if not p.exists():
            logger.error(f"{label} not found: {p}")
            sys.exit(1)

    # ── load reference geometry ───────────────────────────────────────────────
    anchors, ref_frame_idx = _load_annotations(ann_path)
    ref_model = CourtGeometryModel(anchors)
    legal_sign = ref_model.legal_near_sign(
        tuple(anchors.get("legal_ref_near", [960, 800]))
    )
    logger.info(f"Reference model loaded. Legal-side sign: {legal_sign:+d}")

    # ── open video + read reference frame ─────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open: {video_path}")
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

    # ── stabilizer ────────────────────────────────────────────────────────────
    s_cfg = cfg.get("stabilizer", {})
    stabilizer = FrameStabilizer(
        n_features=s_cfg.get("n_features", 4000),
        ratio_test=s_cfg.get("ratio_test", 0.75),
        min_matches=s_cfg.get("min_matches", 15),
        ransac_threshold_px=s_cfg.get("ransac_threshold_px", 4.0),
        top_mask_frac=s_cfg.get("top_mask_frac", 0.20),
        transform_type=s_cfg.get("transform_type", "homography"),
    )
    stabilizer.set_reference(ref_frame)

    ref_cfg = cfg.get("refinement", {})
    do_refine = ref_cfg.get("enabled", True)
    refine_search_px = ref_cfg.get("search_px", 15)
    refine_n_pts = ref_cfg.get("n_sample_points", 40)

    out_cfg = cfg.get("output", {})
    debug_indices = set(out_cfg.get("debug_frame_indices", []))
    draw_anchors = out_cfg.get("draw_anchors", True)

    # ── first pass: all frames ─────────────────────────────────────────────────
    logger.info(f"Processing {total_frames} frames …")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows: list[dict] = []
    prev_H = np.eye(3, dtype=np.float64)
    n_ok = 0
    n_fallback = 0

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

        # Warp court model to current frame
        cur_model = ref_model.warp(H_mat)

        # Optional refinement of kitchen lines
        near_refine, far_refine = 0, 0
        if do_refine and not fallback:
            near_line = cur_model.near_kitchen_line
            far_line = cur_model.far_kitchen_line

            nr_p1, nr_p2, near_refine = _apply_refinement(
                frame, near_line, refine_search_px, refine_n_pts, src_W, src_H
            )
            fr_p1, fr_p2, far_refine = _apply_refinement(
                frame, far_line, refine_search_px, refine_n_pts, src_W, src_H
            )
            # Patch kitchen anchors with refined positions
            refined_anchors = cur_model.anchor_dict()
            refined_anchors["kitchen_near_left"] = list(nr_p1)
            refined_anchors["kitchen_near_right"] = list(nr_p2)
            refined_anchors["kitchen_far_left"] = list(fr_p1)
            refined_anchors["kitchen_far_right"] = list(fr_p2)
            try:
                cur_model = CourtGeometryModel(refined_anchors)
            except Exception:
                pass  # keep unrefined if patching fails

        # Build CSV row
        H_flat = H_mat.flatten().tolist()
        a = cur_model.anchor_dict()
        kp = cur_model.kitchen_endpoints()

        row: dict = {
            "frame_index": frame_idx,
            "timestamp_s": round(ts, 4),
            "H00": H_flat[0], "H01": H_flat[1], "H02": H_flat[2],
            "H10": H_flat[3], "H11": H_flat[4], "H12": H_flat[5],
            "H20": H_flat[6], "H21": H_flat[7], "H22": H_flat[8],
            "n_matches": info["n_matches"],
            "n_inliers": info["n_inliers"],
            "status": info["status"],
            "fallback": int(fallback),
            "near_left_x": round(a["near_left"][0], 2),
            "near_left_y": round(a["near_left"][1], 2),
            "near_right_x": round(a["near_right"][0], 2),
            "near_right_y": round(a["near_right"][1], 2),
            "net_left_x": round(a["net_left"][0], 2),
            "net_left_y": round(a["net_left"][1], 2),
            "net_right_x": round(a["net_right"][0], 2),
            "net_right_y": round(a["net_right"][1], 2),
            "kitchen_near_p1_x": round(kp["near"][0][0], 2),
            "kitchen_near_p1_y": round(kp["near"][0][1], 2),
            "kitchen_near_p2_x": round(kp["near"][1][0], 2),
            "kitchen_near_p2_y": round(kp["near"][1][1], 2),
            "kitchen_far_p1_x": round(kp["far"][0][0], 2),
            "kitchen_far_p1_y": round(kp["far"][0][1], 2),
            "kitchen_far_p2_x": round(kp["far"][1][0], 2),
            "kitchen_far_p2_y": round(kp["far"][1][1], 2),
            "near_refine_offset_px": near_refine,
            "far_refine_offset_px": far_refine,
        }
        rows.append(row)

        # Debug frame
        if frame_idx in debug_indices:
            annotated = draw_court_model(
                frame, cur_model, legal_sign=legal_sign,
                draw_anchors=draw_anchors, fallback=fallback
            )
            annotated = _draw_info_v3(
                annotated, frame_idx, ts,
                info["n_matches"], info["n_inliers"], info["status"], fallback
            )
            dbg_path = debug_dir / f"frame_{frame_idx:05d}.png"
            cv2.imwrite(str(dbg_path), annotated)
            logger.info(f"  Debug frame: {dbg_path}")

        if frame_idx % 500 == 0:
            logger.info(
                f"  … frame {frame_idx}/{total_frames}  "
                f"ok={n_ok}  fallback={n_fallback}"
            )
        frame_idx += 1

    cap.release()
    logger.info(f"First pass: {n_ok} registered, {n_fallback} fallbacks")

    # ── write CSV ─────────────────────────────────────────────────────────────
    csv_path = results_dir / "per_frame_transforms.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"CSV: {csv_path}  ({len(rows)} rows)")

    # ── overlay video ─────────────────────────────────────────────────────────
    if out_cfg.get("save_overlay_video", True):
        overlay_path = results_dir / "overlay.mp4"
        out_fps = float(out_cfg.get("overlay_video_fps", 10.0))
        scale = float(out_cfg.get("overlay_video_scale", 0.5))
        frame_step = int(out_cfg.get("overlay_frame_step", 6))
        out_W = int(src_W * scale)
        out_H = int(src_H * scale)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, out_fps, (out_W, out_H))

        cap2 = cv2.VideoCapture(str(video_path))
        written = 0
        fidx2 = 0
        logger.info("Writing overlay video …")
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            if fidx2 % frame_step == 0 and fidx2 < len(rows):
                r = rows[fidx2]
                # Reconstruct scaled model for this frame
                scaled_anchors = {
                    "near_left": [r["near_left_x"] * scale, r["near_left_y"] * scale],
                    "near_right": [r["near_right_x"] * scale, r["near_right_y"] * scale],
                    "net_left": [r["net_left_x"] * scale, r["net_left_y"] * scale],
                    "net_right": [r["net_right_x"] * scale, r["net_right_y"] * scale],
                    # far corners: warp from reference model at scale
                    "far_left": [
                        ref_model.far_left[0] * scale, ref_model.far_left[1] * scale
                    ],
                    "far_right": [
                        ref_model.far_right[0] * scale, ref_model.far_right[1] * scale
                    ],
                    "kitchen_near_left": [
                        r["kitchen_near_p1_x"] * scale, r["kitchen_near_p1_y"] * scale
                    ],
                    "kitchen_near_right": [
                        r["kitchen_near_p2_x"] * scale, r["kitchen_near_p2_y"] * scale
                    ],
                    "kitchen_far_left": [
                        r["kitchen_far_p1_x"] * scale, r["kitchen_far_p1_y"] * scale
                    ],
                    "kitchen_far_right": [
                        r["kitchen_far_p2_x"] * scale, r["kitchen_far_p2_y"] * scale
                    ],
                }
                small = cv2.resize(frame, (out_W, out_H))
                try:
                    frame_model = CourtGeometryModel(scaled_anchors)
                    small = draw_court_model(
                        small, frame_model, legal_sign=legal_sign,
                        draw_anchors=draw_anchors, fallback=bool(r["fallback"])
                    )
                except Exception:
                    pass
                small = _draw_info_v3(
                    small, fidx2, r["timestamp_s"],
                    r["n_matches"], r["n_inliers"], r["status"], bool(r["fallback"])
                )
                writer.write(small)
                written += 1
            fidx2 += 1
        cap2.release()
        writer.release()
        logger.info(f"Overlay video: {overlay_path}  ({written} frames)")

    # ── validation report ─────────────────────────────────────────────────────
    val_cfg = cfg.get("validation", {})
    n_sample = min(val_cfg.get("n_sample_frames", 60), len(rows))
    sample_indices = np.linspace(0, len(rows) - 1, n_sample, dtype=int)

    cap3 = cv2.VideoCapture(str(video_path))
    near_strengths, far_strengths, translations = [], [], []
    for si in sample_indices:
        r = rows[si]
        cap3.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame_index"]))
        ret, frm = cap3.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        p1 = (r["kitchen_near_p1_x"], r["kitchen_near_p1_y"])
        p2 = (r["kitchen_near_p2_x"], r["kitchen_near_p2_y"])
        near_strengths.append(_edge_strength(gray, p1, p2))
        p1f = (r["kitchen_far_p1_x"], r["kitchen_far_p1_y"])
        p2f = (r["kitchen_far_p2_x"], r["kitchen_far_p2_y"])
        far_strengths.append(_edge_strength(gray, p1f, p2f))
        tx, ty = abs(r["H02"]), abs(r["H12"])
        translations.append(float(np.sqrt(tx * tx + ty * ty)))
    cap3.release()

    def _stats(arr):
        if not arr:
            return {}
        a = np.array(arr)
        return {
            "mean": round(float(a.mean()), 2),
            "std": round(float(a.std()), 2),
            "cv": round(float(a.std() / (a.mean() + 1e-6)), 4),
            "min": round(float(a.min()), 2),
            "max": round(float(a.max()), 2),
            "n": len(arr),
        }

    near_stats = _stats(near_strengths)
    far_stats = _stats(far_strengths)
    trans_stats = _stats(translations)

    overall = (
        "stable"
        if near_stats.get("cv", 1.0) < 0.20 and far_stats.get("cv", 1.0) < 0.20
        else "check"
    )

    validation = {
        "near_kitchen_edge_strength": near_stats,
        "far_kitchen_edge_strength": far_stats,
        "homography_translation_px": trans_stats,
        "overall_assessment": overall,
        "n_frames_sampled": len(near_strengths),
    }

    # ── summary report ────────────────────────────────────────────────────────
    summary = {
        "run_name": run_name,
        "video": video_path.name,
        "resolution": f"{src_W}x{src_H}",
        "fps": round(src_fps, 3),
        "total_frames": total_frames,
        "duration_s": round(duration_s, 2),
        "method": "anchor-point court model + ORB homography",
        "reference_frame_index": ref_frame_idx,
        "annotation_source": str(ann_path),
        "reference_anchors": anchors,
        "registration": {
            "n_ok": n_ok,
            "n_fallback": n_fallback,
            "fallback_rate": round(n_fallback / max(1, total_frames), 4),
        },
        "validation": validation,
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

    # ── console summary ───────────────────────────────────────────────────────
    print("\n── court_reg_v3 results ──────────────────────────────────────────")
    print(f"  Registration:  {n_ok} ok  {n_fallback} fallback  "
          f"({n_fallback / max(1, total_frames) * 100:.1f}% fallback)")
    print(f"  Near kitchen edge strength:  "
          f"mean={near_stats.get('mean')}  cv={near_stats.get('cv')}")
    print(f"  Far kitchen edge strength:   "
          f"mean={far_stats.get('mean')}  cv={far_stats.get('cv')}")
    print(f"  Homography translation (px): "
          f"mean={trans_stats.get('mean')}  max={trans_stats.get('max')}")
    print(f"  Overall: {overall}")
    print()
    print("  Next: if anchors are off, re-run:")
    print(f"    python scripts/annotate_anchors.py \\")
    print(f"        --video {video_path} \\")
    print(f"        --frame {ref_frame_idx} \\")
    print(f"        --out   {ann_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Court registration v3 (anchor-point model + ORB homography)"
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/court_reg_v3.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    run(Path(args.config))
