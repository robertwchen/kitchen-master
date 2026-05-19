"""
Phase 1 v3 — Court Registration from Anchor Points + ORB Transform Estimation.

Strategy
--------
1. User annotates 6–11 anchor points on a clean reference frame:
   near corners, far corners, net ends, optional kitchen-line intersections.
2. CourtGeometryModel derives the full court structure (net, kitchen lines,
   legal zone polygons) from those anchors.
3. ORB + BFMatcher + RANSAC estimates an affine transform or homography
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
import copy
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


def _stats(arr: list[float]) -> dict:
    if not arr:
        return {}
    a = np.array(arr, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 2),
        "median": round(float(np.median(a)), 2),
        "std": round(float(a.std()), 2),
        "cv": round(float(a.std() / (a.mean() + 1e-6)), 4),
        "min": round(float(a.min()), 2),
        "max": round(float(a.max()), 2),
        "n": len(arr),
    }


def _expand_quad(quad: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    center = quad.mean(axis=0)
    out = quad.copy()
    out[:, 0] = center[0] + (out[:, 0] - center[0]) * scale_x
    out[:, 1] = center[1] + (out[:, 1] - center[1]) * scale_y
    return out


def _build_feature_roi_mask(
    ref_model: CourtGeometryModel,
    height: int,
    width: int,
    roi_cfg: dict,
) -> np.ndarray | None:
    if not roi_cfg.get("enabled", True):
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    kp = ref_model.kitchen_endpoints()
    line_band_px = int(roi_cfg.get("line_band_px", 72))
    court_padding_px = int(roi_cfg.get("court_padding_px", 28))
    scale_x = float(roi_cfg.get("expand_scale_x", 1.08))
    scale_y = float(roi_cfg.get("expand_scale_y", 1.18))
    fill_court = bool(roi_cfg.get("fill_court", False))

    if fill_court and "far" in kp:
        quad = np.array(
            [kp["near"][0], kp["near"][1], kp["far"][1], kp["far"][0]],
            dtype=np.float32,
        )
        quad = _expand_quad(quad, scale_x, scale_y)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    elif fill_court:
        cv2.line(mask, _ipt(kp["near"][0]), _ipt(kp["near"][1]), 255, line_band_px * 2)

    for line in (
        ref_model.near_kitchen_line,
        ref_model.far_kitchen_line,
        ref_model.left_boundary_line,
        ref_model.right_boundary_line,
    ):
        if line is None:
            continue
        pt1, pt2 = line.endpoints_in_frame(width, height)
        cv2.line(mask, pt1, pt2, 255, line_band_px)

    if court_padding_px > 0:
        k = court_padding_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def _default_tracking_point(ref_model: CourtGeometryModel) -> tuple[float, float]:
    if ref_model.left_boundary_line is not None and ref_model.right_boundary_line is not None:
        kn = ref_model.kitchen_endpoints()["near"]
        return (
            float((kn[0][0] + kn[1][0]) / 2.0),
            float((kn[0][1] + kn[1][1]) / 2.0),
        )
    kn = ref_model.kitchen_endpoints()["near"]
    return (
        float((kn[0][0] + kn[1][0]) / 2.0),
        float((kn[0][1] + kn[1][1]) / 2.0),
    )


def _resolve_tracking_point(
    tracker_cfg: dict,
    ref_model: CourtGeometryModel,
) -> tuple[float, float]:
    if tracker_cfg.get("reference_point") is not None:
        pt = tracker_cfg["reference_point"]
        return float(pt[0]), float(pt[1])

    annotations_path = tracker_cfg.get("annotations_path")
    anchor_name = tracker_cfg.get("anchor_name", "net_base_center")
    if annotations_path:
        ann_path = Path(annotations_path)
        if ann_path.exists():
            with open(ann_path) as f:
                data = json.load(f)
            frames = data.get("annotated_frames", [])
            if frames:
                frame0 = frames[0]
                if "anchors" in frame0 and anchor_name in frame0["anchors"]:
                    pt = frame0["anchors"][anchor_name]
                    return float(pt[0]), float(pt[1])
                if "anchor_points" in frame0 and anchor_name in frame0["anchor_points"]:
                    pt = frame0["anchor_points"][anchor_name]
                    return float(pt[0]), float(pt[1])

    return _default_tracking_point(ref_model)


def _extract_template_patch(
    gray: np.ndarray,
    center: tuple[float, float],
    half_size: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0 = max(0, cx - half_size)
    y0 = max(0, cy - half_size)
    x1 = min(gray.shape[1], cx + half_size + 1)
    y1 = min(gray.shape[0], cy + half_size + 1)
    patch = gray[y0:y1, x0:x1].copy()
    if patch.size == 0:
        raise ValueError("Template patch is empty; check tracking reference point")
    return patch, (x0, y0)


def _estimate_post_translation(
    gray: np.ndarray,
    template: np.ndarray,
    search_center: tuple[float, float],
    search_radius: int,
    min_score: float,
) -> tuple[tuple[float, float] | None, dict]:
    th, tw = template.shape[:2]
    cx, cy = int(round(search_center[0])), int(round(search_center[1]))
    x0 = max(0, cx - search_radius - tw // 2)
    y0 = max(0, cy - search_radius - th // 2)
    x1 = min(gray.shape[1], cx + search_radius + tw // 2 + 1)
    y1 = min(gray.shape[0], cy + search_radius + th // 2 + 1)
    roi = gray[y0:y1, x0:x1]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return None, {"status": "search_window_too_small", "score": 0.0}

    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if float(max_val) < min_score:
        return None, {"status": "template_score_low", "score": float(max_val)}

    top_left = (x0 + max_loc[0], y0 + max_loc[1])
    center = (top_left[0] + tw / 2.0, top_left[1] + th / 2.0)
    return center, {"status": "ok", "score": float(max_val)}


def _load_validation_entries(labels_path: Path) -> tuple[dict, list[dict]]:
    with open(labels_path) as f:
        data = json.load(f)
    entries = data.get("frames", data.get("annotated_frames", []))
    return data, entries


def _compute_reprojection_report(
    labels_path: Path,
    rows: list[dict],
    ref_model: CourtGeometryModel,
) -> dict | None:
    labels_data, entries = _load_validation_entries(labels_path)
    row_by_frame = {int(r["frame_index"]): r for r in rows}
    ref_anchors = ref_model.anchor_dict()

    per_anchor_errors: dict[str, list[float]] = {}
    all_errors: list[float] = []
    compared_frames = 0

    for entry in entries:
        fidx = int(entry["frame_index"])
        labeled = entry.get("anchors", {})
        row = row_by_frame.get(fidx)
        if row is None:
            continue
        compared_frames += 1
        H = np.array(
            [
                [row["H00"], row["H01"], row["H02"]],
                [row["H10"], row["H11"], row["H12"]],
                [row["H20"], row["H21"], row["H22"]],
            ],
            dtype=np.float64,
        )
        for key, ref_pt in ref_anchors.items():
            if key not in labeled:
                continue
            p = np.array([[[float(ref_pt[0]), float(ref_pt[1])]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(p, H)
            px, py = float(proj[0, 0, 0]), float(proj[0, 0, 1])
            lx, ly = float(labeled[key][0]), float(labeled[key][1])
            err = float(np.hypot(px - lx, py - ly))
            per_anchor_errors.setdefault(key, []).append(err)
            all_errors.append(err)

    if not all_errors:
        return None

    return {
        "labels_path": str(labels_path),
        "n_labeled_frames": len(entries),
        "n_compared_frames": compared_frames,
        "overall": {
            "mean": round(float(np.mean(all_errors)), 2),
            "median": round(float(np.median(all_errors)), 2),
            "max": round(float(np.max(all_errors)), 2),
            "n": len(all_errors),
        },
        "per_anchor": {
            key: {
                "mean": round(float(np.mean(vals)), 2),
                "median": round(float(np.median(vals)), 2),
                "max": round(float(np.max(vals)), 2),
                "n": len(vals),
            }
            for key, vals in per_anchor_errors.items()
            if vals
        },
    }


def _run_reprojection_validation(
    labels_path: Path,
    rows: list[dict],
    ref_model: CourtGeometryModel,
    results_dir: Path,
) -> dict | None:
    report = _compute_reprojection_report(labels_path, rows, ref_model)
    if report is None:
        logger.warning("Reprojection validation: no matching labeled frames found.")
        return None

    rp = results_dir / "reprojection_errors.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)

    print("\n── reprojection validation ───────────────────────────────────────")
    ov = report["overall"]
    print(f"  Overall: mean={ov['mean']}px  median={ov['median']}px  max={ov['max']}px  (n={ov['n']})")
    for k, s in report["per_anchor"].items():
        print(f"  {k:25s}: mean={s['mean']}px  max={s['max']}px")
    logger.info(f"Reprojection report: {rp}")
    return report


def _make_variant_cfg(
    base_cfg: dict,
    *,
    transform_type: str | None = None,
    rolling_reference: bool | None = None,
    refinement_enabled: bool | None = None,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if transform_type is not None:
        cfg.setdefault("stabilizer", {})["transform_type"] = transform_type
    if rolling_reference is not None:
        cfg.setdefault("stabilizer", {})["rolling_reference"] = rolling_reference
    if refinement_enabled is not None:
        cfg.setdefault("refinement", {})["enabled"] = refinement_enabled
    return cfg


def _variant_key(cfg: dict) -> tuple[str, bool, bool]:
    s_cfg = cfg.get("stabilizer", {})
    r_cfg = cfg.get("refinement", {})
    return (
        str(s_cfg.get("transform_type", "homography")),
        bool(s_cfg.get("rolling_reference", False)),
        bool(r_cfg.get("enabled", True)),
    )


def _variant_label(cfg: dict) -> str:
    s_cfg = cfg.get("stabilizer", {})
    r_cfg = cfg.get("refinement", {})
    mode = str(s_cfg.get("transform_type", "homography"))
    ref_mode = "rolling" if s_cfg.get("rolling_reference", False) else "fixed"
    refine = "refine-on" if r_cfg.get("enabled", True) else "refine-off"
    return f"{mode} | {ref_mode} | {refine}"


def _model_from_row(row: dict, scale: float = 1.0) -> CourtGeometryModel:
    anchors = {
        "kitchen_near_left": [
            row["kitchen_near_p1_x"] * scale,
            row["kitchen_near_p1_y"] * scale,
        ],
        "kitchen_near_right": [
            row["kitchen_near_p2_x"] * scale,
            row["kitchen_near_p2_y"] * scale,
        ],
    }
    if row.get("kitchen_far_p1_x") is not None:
        anchors["kitchen_far_left"] = [
            row["kitchen_far_p1_x"] * scale,
            row["kitchen_far_p1_y"] * scale,
        ]
        anchors["kitchen_far_right"] = [
            row["kitchen_far_p2_x"] * scale,
            row["kitchen_far_p2_y"] * scale,
        ]
    return CourtGeometryModel(anchors)


def _render_variant_frame(
    frame: np.ndarray,
    row: dict,
    label: str,
    draw_anchors: bool,
) -> np.ndarray:
    out = frame.copy()
    try:
        model = _model_from_row(row)
        out = draw_court_model(
            out,
            model,
            draw_anchors=draw_anchors,
            fallback=bool(row["fallback"]),
        )
    except Exception:
        pass
    out = _draw_info_v3(
        out,
        int(row["frame_index"]),
        float(row["timestamp_s"]),
        int(row["n_matches"]),
        int(row["n_inliers"]),
        str(row["status"]),
        bool(row["fallback"]),
    )
    cv2.putText(out, label, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(out, label, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def _write_side_by_side_debug_frames(
    video_path: Path,
    out_dir: Path,
    debug_indices: list[int],
    left_rows: list[dict],
    left_label: str,
    right_rows: list[dict],
    right_label: str,
    draw_anchors: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    left_by_frame = {int(r["frame_index"]): r for r in left_rows}
    right_by_frame = {int(r["frame_index"]): r for r in right_rows}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video for comparison export: {video_path}")
        return

    for frame_idx in debug_indices:
        left_row = left_by_frame.get(frame_idx)
        right_row = right_by_frame.get(frame_idx)
        if left_row is None or right_row is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        left_img = _render_variant_frame(frame, left_row, left_label, draw_anchors)
        right_img = _render_variant_frame(frame, right_row, right_label, draw_anchors)
        combo = np.hstack([left_img, right_img])
        out_path = out_dir / f"frame_{frame_idx:05d}.png"
        cv2.imwrite(str(out_path), combo)
    cap.release()


def _process_variant(
    video_path: Path,
    start_frame: int,
    ref_model: CourtGeometryModel,
    ref_frame: np.ndarray,
    cfg: dict,
    debug_indices: set[int],
    draw_anchors: bool,
    debug_dir: Path | None = None,
    roi_mask: np.ndarray | None = None,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / src_fps

    s_cfg = cfg.get("stabilizer", {})
    r_cfg = cfg.get("refinement", {})
    transform_mode = str(s_cfg.get("transform_type", "homography"))
    rolling_reference = bool(s_cfg.get("rolling_reference", False))
    do_refine = bool(r_cfg.get("enabled", True))
    refine_search_px = int(r_cfg.get("search_px", 15))
    refine_n_pts = int(r_cfg.get("n_sample_points", 40))

    stabilizer = None
    tracker_cfg = s_cfg.get("translation_tracker", {})
    template = None
    template_center = None
    template_search_center = None
    template_use_previous = bool(tracker_cfg.get("use_previous_match", True))
    template_search_radius = int(tracker_cfg.get("search_radius_px", 36))
    template_min_score = float(tracker_cfg.get("min_score", 0.55))

    if transform_mode in {"affine", "homography"}:
        stabilizer = FrameStabilizer(
            n_features=s_cfg.get("n_features", 4000),
            ratio_test=s_cfg.get("ratio_test", 0.75),
            min_matches=s_cfg.get("min_matches", 15),
            ransac_threshold_px=s_cfg.get("ransac_threshold_px", 4.0),
            top_mask_frac=s_cfg.get("top_mask_frac", 0.20),
            bottom_mask_frac=s_cfg.get("bottom_mask_frac", 0.0),
            transform_type=transform_mode,
            max_translation_px=s_cfg.get("max_translation_px", 80.0),
            max_det_dev=s_cfg.get("max_det_dev", 0.25),
            max_rotation_deg=s_cfg.get("max_rotation_deg"),
            max_scale_dev=s_cfg.get("max_scale_dev"),
        )
        stabilizer.set_feature_mask(roi_mask)
        stabilizer.set_reference(ref_frame)
    elif transform_mode == "post_translation":
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        template_center = _resolve_tracking_point(tracker_cfg, ref_model)
        template, _ = _extract_template_patch(
            ref_gray,
            template_center,
            int(tracker_cfg.get("template_half_size_px", 24)),
        )
        template_search_center = template_center
    elif transform_mode != "static":
        raise ValueError(f"Unknown transform_type: {transform_mode}")

    rows: list[dict] = []
    H_cumulative = np.eye(3, dtype=np.float64)
    prev_H = np.eye(3, dtype=np.float64)
    n_ok = 0
    n_fallback = 0
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / src_fps
        if transform_mode == "static":
            H_mat = np.eye(3, dtype=np.float64)
            info = {
                "n_matches": 0,
                "n_inliers": 0,
                "status": "static_reference",
                "score": 1.0,
            }
            fallback = False
            n_ok += 1
        elif transform_mode == "post_translation":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            match_center, match_info = _estimate_post_translation(
                gray,
                template,
                template_search_center,
                template_search_radius,
                template_min_score,
            )
            fallback = match_center is None
            if fallback:
                H_mat = prev_H
                n_fallback += 1
                info = {
                    "n_matches": 1,
                    "n_inliers": 0,
                    "status": match_info["status"],
                    "score": round(match_info.get("score", 0.0), 4),
                }
            else:
                dx = float(match_center[0] - template_center[0])
                dy = float(match_center[1] - template_center[1])
                H_mat = np.array(
                    [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                prev_H = H_mat
                if template_use_previous:
                    template_search_center = match_center
                info = {
                    "n_matches": 1,
                    "n_inliers": 1,
                    "status": "template_ok",
                    "score": round(match_info.get("score", 0.0), 4),
                }
                n_ok += 1
        else:
            H_rel, info = stabilizer.estimate_transform(
                frame,
                update_ref_on_success=rolling_reference,
            )
            fallback = H_rel is None
            if fallback:
                H_mat = prev_H
                n_fallback += 1
            else:
                if rolling_reference:
                    H_cumulative = H_rel @ H_cumulative
                    H_mat = H_cumulative
                else:
                    H_mat = H_rel
                prev_H = H_mat
                n_ok += 1

        if frame_idx < start_frame:
            frame_idx += 1
            continue

        cur_model = ref_model.warp(H_mat)

        near_refine, far_refine = 0, 0
        if do_refine and not fallback:
            refined_anchors = cur_model.anchor_dict()
            nr_p1, nr_p2, near_refine = _apply_refinement(
                frame,
                cur_model.near_kitchen_line,
                refine_search_px,
                refine_n_pts,
                src_W,
                src_H,
            )
            refined_anchors["kitchen_near_left"] = list(nr_p1)
            refined_anchors["kitchen_near_right"] = list(nr_p2)
            if cur_model.far_kitchen_line is not None:
                fr_p1, fr_p2, far_refine = _apply_refinement(
                    frame,
                    cur_model.far_kitchen_line,
                    refine_search_px,
                    refine_n_pts,
                    src_W,
                    src_H,
                )
                refined_anchors["kitchen_far_left"] = list(fr_p1)
                refined_anchors["kitchen_far_right"] = list(fr_p2)
            try:
                cur_model = CourtGeometryModel(refined_anchors)
            except Exception:
                pass

        kp = cur_model.kitchen_endpoints()
        has_far = "far" in kp
        H_flat = H_mat.flatten().tolist()
        row = {
            "frame_index": frame_idx,
            "timestamp_s": round(ts, 4),
            "H00": H_flat[0],
            "H01": H_flat[1],
            "H02": H_flat[2],
            "H10": H_flat[3],
            "H11": H_flat[4],
            "H12": H_flat[5],
            "H20": H_flat[6],
            "H21": H_flat[7],
            "H22": H_flat[8],
            "n_matches": info["n_matches"],
            "n_inliers": info["n_inliers"],
            "status": info["status"],
            "fallback": int(fallback),
            "transform_type": transform_mode,
            "rolling_reference": int(rolling_reference),
            "refinement_enabled": int(do_refine),
            "tracker_score": round(float(info.get("score", 0.0)), 4),
            "kitchen_near_p1_x": round(kp["near"][0][0], 2),
            "kitchen_near_p1_y": round(kp["near"][0][1], 2),
            "kitchen_near_p2_x": round(kp["near"][1][0], 2),
            "kitchen_near_p2_y": round(kp["near"][1][1], 2),
            "kitchen_far_p1_x": round(kp["far"][0][0], 2) if has_far else None,
            "kitchen_far_p1_y": round(kp["far"][0][1], 2) if has_far else None,
            "kitchen_far_p2_x": round(kp["far"][1][0], 2) if has_far else None,
            "kitchen_far_p2_y": round(kp["far"][1][1], 2) if has_far else None,
            "near_refine_offset_px": near_refine,
            "far_refine_offset_px": far_refine,
        }
        rows.append(row)

        if debug_dir is not None and frame_idx in debug_indices:
            annotated = draw_court_model(
                frame,
                cur_model,
                draw_anchors=draw_anchors,
                fallback=fallback,
            )
            annotated = _draw_info_v3(
                annotated,
                frame_idx,
                ts,
                info["n_matches"],
                info["n_inliers"],
                info["status"],
                fallback,
            )
            dbg_path = debug_dir / f"frame_{frame_idx:05d}.png"
            cv2.imwrite(str(dbg_path), annotated)
            logger.info(f"  Debug frame: {dbg_path}")

        frame_idx += 1

    cap.release()
    return {
        "rows": rows,
        "n_ok": n_ok,
        "n_fallback": n_fallback,
        "src_fps": src_fps,
        "total_frames": total_frames,
        "src_W": src_W,
        "src_H": src_H,
        "duration_s": duration_s,
        "variant_label": _variant_label(cfg),
        "settings": {
            "transform_type": transform_mode,
            "rolling_reference": rolling_reference,
            "refinement_enabled": do_refine,
        },
    }


def _write_overlay_video(
    video_path: Path,
    rows: list[dict],
    out_path: Path,
    scale: float,
    fps: float,
    frame_step: int,
    draw_anchors: bool,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video for overlay export: {video_path}")
        return 0

    src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_W = int(src_W * scale)
    out_H = int(src_H * scale)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_W, out_H))
    written = 0

    for row_pos, row in enumerate(rows):
        if row_pos % frame_step != 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
        ret, frame = cap.read()
        if not ret:
            continue
        small = cv2.resize(frame, (out_W, out_H))
        try:
            frame_model = _model_from_row(row, scale=scale)
            small = draw_court_model(
                small,
                frame_model,
                draw_anchors=draw_anchors,
                fallback=bool(row["fallback"]),
            )
        except Exception:
            pass
        small = _draw_info_v3(
            small,
            int(row["frame_index"]),
            float(row["timestamp_s"]),
            int(row["n_matches"]),
            int(row["n_inliers"]),
            str(row["status"]),
            bool(row["fallback"]),
        )
        writer.write(small)
        written += 1

    cap.release()
    writer.release()
    return written


def _build_stability_validation(
    video_path: Path,
    rows: list[dict],
    n_sample_frames: int,
) -> dict:
    if not rows:
        return {
            "left_boundary_edge_strength": {},
            "right_boundary_edge_strength": {},
            "transform_translation_px": {},
            "overall_assessment": "no_rows",
            "n_frames_sampled": 0,
        }

    n_sample = min(n_sample_frames, len(rows))
    sample_indices = np.linspace(0, len(rows) - 1, n_sample, dtype=int)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "left_boundary_edge_strength": {},
            "right_boundary_edge_strength": {},
            "transform_translation_px": {},
            "overall_assessment": "video_unavailable",
            "n_frames_sampled": 0,
        }

    left_strengths, right_strengths, translations = [], [], []
    for si in sample_indices:
        row = rows[int(si)]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
        ret, frm = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        if row.get("kitchen_far_p1_x") is not None:
            left_strengths.append(
                _edge_strength(
                    gray,
                    (row["kitchen_near_p1_x"], row["kitchen_near_p1_y"]),
                    (row["kitchen_far_p1_x"], row["kitchen_far_p1_y"]),
                )
            )
            right_strengths.append(
                _edge_strength(
                    gray,
                    (row["kitchen_near_p2_x"], row["kitchen_near_p2_y"]),
                    (row["kitchen_far_p2_x"], row["kitchen_far_p2_y"]),
                )
            )
        tx, ty = abs(float(row["H02"])), abs(float(row["H12"]))
        translations.append(float(np.hypot(tx, ty)))
    cap.release()

    left_stats = _stats(left_strengths)
    right_stats = _stats(right_strengths)
    trans_stats = _stats(translations)
    overall = (
        "stable"
        if left_stats.get("cv", 1.0) < 0.20 and right_stats.get("cv", 1.0) < 0.20
        else "check"
    )
    return {
        "left_boundary_edge_strength": left_stats,
        "right_boundary_edge_strength": right_stats,
        "transform_translation_px": trans_stats,
        "overall_assessment": overall,
        "n_frames_sampled": len(translations),
    }


def _export_comparison_pairs(
    base_cfg: dict,
    video_path: Path,
    start_frame: int,
    ref_model: CourtGeometryModel,
    ref_frame: np.ndarray,
    debug_indices: list[int],
    draw_anchors: bool,
    roi_mask: np.ndarray | None,
    results_dir: Path,
    labels_path: Path | None,
    primary_cache: dict[tuple[str, bool, bool], dict] | None = None,
) -> dict:
    cmp_dir = results_dir / "comparisons"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    cache = primary_cache or {}

    def get_variant(cfg: dict) -> dict:
        key = _variant_key(cfg)
        if key not in cache:
            cache[key] = _process_variant(
                video_path,
                start_frame,
                ref_model,
                ref_frame,
                cfg,
                set(debug_indices),
                draw_anchors,
                debug_dir=None,
                roi_mask=roi_mask,
            )
            if labels_path is not None and labels_path.exists():
                cache[key]["reprojection"] = _compute_reprojection_report(
                    labels_path,
                    cache[key]["rows"],
                    ref_model,
                )
        return cache[key]

    pair_specs = [
        (
            "post_translation_vs_affine_fixed",
            _make_variant_cfg(base_cfg, transform_type="post_translation", rolling_reference=False),
            _make_variant_cfg(base_cfg, transform_type="affine", rolling_reference=False),
        ),
        (
            "post_translation_vs_static",
            _make_variant_cfg(base_cfg, transform_type="post_translation", rolling_reference=False),
            _make_variant_cfg(base_cfg, transform_type="static", rolling_reference=False),
        ),
        (
            "refinement_on_vs_off",
            _make_variant_cfg(base_cfg, transform_type="affine", rolling_reference=False, refinement_enabled=True),
            _make_variant_cfg(base_cfg, transform_type="affine", rolling_reference=False, refinement_enabled=False),
        ),
    ]

    report: dict = {}
    for pair_name, left_cfg, right_cfg in pair_specs:
        left = get_variant(left_cfg)
        right = get_variant(right_cfg)
        pair_dir = cmp_dir / pair_name
        _write_side_by_side_debug_frames(
            video_path,
            pair_dir,
            debug_indices,
            left["rows"],
            left["variant_label"],
            right["rows"],
            right["variant_label"],
            draw_anchors,
        )
        report[pair_name] = {
            "left": {
                "label": left["variant_label"],
                "settings": left["settings"],
                "registration": {
                    "n_ok": left["n_ok"],
                    "n_fallback": left["n_fallback"],
                },
                "reprojection": left.get("reprojection"),
            },
            "right": {
                "label": right["variant_label"],
                "settings": right["settings"],
                "registration": {
                    "n_ok": right["n_ok"],
                    "n_fallback": right["n_fallback"],
                },
                "reprojection": right.get("reprojection"),
            },
            "output_dir": str(pair_dir),
        }

    report_path = cmp_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Comparison exports: {report_path}")
    return report


# ── main ─────────────────────────────────────────────────────────────────────

def run(config_path: Path) -> None:
    cfg = _load_config(config_path)
    run_name = cfg["run_name"]
    video_path = Path(cfg["video"]["path"])
    start_frame = int(cfg["video"].get("start_frame", 0))
    ann_path = Path(cfg["annotations"]["path"])
    results_dir = Path(cfg["output"]["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = results_dir / "debug_frames"
    debug_dir.mkdir(exist_ok=True)

    for p, label in [(video_path, "video"), (ann_path, "annotations")]:
        if not p.exists():
            logger.error(f"{label} not found: {p}")
            sys.exit(1)

    anchors, ref_frame_idx = _load_annotations(ann_path)
    ref_model = CourtGeometryModel(anchors)
    has_boundaries = ref_model.left_boundary_line is not None
    logger.info(
        f"Reference model loaded. "
        f"Boundaries: {'left+right NVZ lines' if has_boundaries else 'near kitchen only'}"
    )

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
    cap.release()
    if not ret:
        logger.error(f"Cannot read reference frame {ref_frame_idx}")
        sys.exit(1)

    s_cfg = cfg.get("stabilizer", {})
    out_cfg = cfg.get("output", {})
    val_cfg = cfg.get("validation", {})
    debug_indices = set(out_cfg.get("debug_frame_indices", []))
    draw_anchors = out_cfg.get("draw_anchors", True)

    logger.info(
        f"Processing {total_frames} frames  "
        f"(recording from frame {start_frame}, t={start_frame/src_fps:.1f}s) …"
    )

    roi_mask = _build_feature_roi_mask(
        ref_model,
        src_H,
        src_W,
        s_cfg.get("roi", {}),
    )
    roi_mask_path = None
    if roi_mask is not None:
        roi_mask_path = results_dir / "feature_roi_mask.png"
        cv2.imwrite(str(roi_mask_path), roi_mask)
        logger.info(f"Feature ROI mask: {roi_mask_path}")

    primary = _process_variant(
        video_path,
        start_frame,
        ref_model,
        ref_frame,
        cfg,
        debug_indices,
        draw_anchors,
        debug_dir=debug_dir,
        roi_mask=roi_mask,
    )
    rows = primary["rows"]
    n_ok = primary["n_ok"]
    n_fallback = primary["n_fallback"]
    logger.info(f"First pass: {n_ok} registered, {n_fallback} fallbacks")

    csv_path = results_dir / "per_frame_transforms.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"CSV: {csv_path}  ({len(rows)} rows)")

    overlay_path = results_dir / "overlay.mp4"
    if out_cfg.get("save_overlay_video", True):
        written = _write_overlay_video(
            video_path,
            rows,
            overlay_path,
            scale=float(out_cfg.get("overlay_video_scale", 0.5)),
            fps=float(out_cfg.get("overlay_video_fps", 10.0)),
            frame_step=int(out_cfg.get("overlay_frame_step", 6)),
            draw_anchors=draw_anchors,
        )
        logger.info(f"Overlay video: {overlay_path}  ({written} frames)")

    validation = _build_stability_validation(
        video_path,
        rows,
        int(val_cfg.get("n_sample_frames", 60)),
    )

    reprojection_report = None
    reprojection_labels_path = val_cfg.get("reprojection_labels_path")
    labels_path = Path(reprojection_labels_path) if reprojection_labels_path else None
    if labels_path is not None:
        if labels_path.exists():
            reprojection_report = _run_reprojection_validation(
                labels_path,
                rows,
                ref_model,
                results_dir,
            )
        else:
            logger.warning(f"Reprojection labels not found: {labels_path}")

    summary = {
        "run_name": run_name,
        "video": video_path.name,
        "resolution": f"{src_W}x{src_H}",
        "fps": round(src_fps, 3),
        "total_frames": total_frames,
        "duration_s": round(duration_s, 2),
        "method": f"anchor-point court model + ORB {primary['settings']['transform_type']}",
        "reference_frame_index": ref_frame_idx,
        "annotation_source": str(ann_path),
        "reference_anchors": anchors,
        "stabilizer": cfg.get("stabilizer", {}),
        "refinement": cfg.get("refinement", {}),
        "registration": {
            "n_ok": n_ok,
            "n_fallback": n_fallback,
            "fallback_rate": round(n_fallback / max(1, total_frames), 4),
            "rolling_reference": primary["settings"]["rolling_reference"],
        },
        "validation": validation,
        "reprojection_validation": reprojection_report,
        "outputs": {
            "per_frame_transforms_csv": str(csv_path),
            "debug_frames": str(debug_dir),
            "overlay_video": str(overlay_path),
        },
    }
    if roi_mask_path is not None:
        summary["outputs"]["feature_roi_mask"] = str(roi_mask_path)

    comparison_exports = None
    cmp_cfg = cfg.get("comparison_exports", {})
    if cmp_cfg.get("enabled", True):
        primary_cache = {_variant_key(cfg): dict(primary)}
        if reprojection_report is not None:
            primary_cache[_variant_key(cfg)]["reprojection"] = reprojection_report
        comparison_exports = _export_comparison_pairs(
            base_cfg=cfg,
            video_path=video_path,
            start_frame=start_frame,
            ref_model=ref_model,
            ref_frame=ref_frame,
            debug_indices=sorted(debug_indices),
            draw_anchors=draw_anchors,
            roi_mask=roi_mask,
            results_dir=results_dir,
            labels_path=labels_path if labels_path and labels_path.exists() else None,
            primary_cache=primary_cache,
        )
        summary["outputs"]["comparison_exports"] = str(results_dir / "comparisons")
        summary["comparison_exports"] = comparison_exports

    summary_path = results_dir / "summary_report.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary_path}")

    left_stats = validation.get("left_boundary_edge_strength", {})
    right_stats = validation.get("right_boundary_edge_strength", {})
    trans_stats = validation.get("transform_translation_px", {})
    overall = validation.get("overall_assessment")

    print("\n── court_reg_v3 results ──────────────────────────────────────────")
    print(
        f"  Registration:  {n_ok} ok  {n_fallback} fallback  "
        f"({n_fallback / max(1, total_frames) * 100:.1f}% fallback)"
    )
    print(
        f"  Left  boundary edge strength: "
        f"mean={left_stats.get('mean')}  cv={left_stats.get('cv')}"
    )
    print(
        f"  Right boundary edge strength: "
        f"mean={right_stats.get('mean')}  cv={right_stats.get('cv')}"
    )
    print(
        f"  Transform translation (px): "
        f"mean={trans_stats.get('mean')}  max={trans_stats.get('max')}"
    )
    print(f"  Overall: {overall}")
    if comparison_exports:
        print("  Comparison exports: results/real_baseline/court_reg_v3/comparisons/")
    print()
    print("  Next: if anchors are off, re-run:")
    print(f"    python scripts/annotate_anchors.py \\")
    print(f"        --video {video_path} \\")
    print(f"        --frame {ref_frame_idx} \\")
    print(f"        --out   {ann_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Court registration v3 (anchor-point model + ORB transform)"
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/court_reg_v3.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    run(Path(args.config))
