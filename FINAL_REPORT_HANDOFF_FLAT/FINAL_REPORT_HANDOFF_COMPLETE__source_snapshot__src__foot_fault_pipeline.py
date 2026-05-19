"""
Foot-fault decision pipeline.

For each volley event:
  1. Load the registered NVZ boundary for that frame (from registration CSV or
     manual override).
  2. Localize the relevant foot using foot_localizer.
  3. Compute signed distance from foot point to the NVZ boundary line.
  4. Classify:
       legal_volley       — foot clearly behind the line (dist > +threshold)
       foot_fault_volley  — foot on or over the line (dist < -threshold)
       uncertain          — within ±uncertain_margin_px of the line

Verification artifacts
----------------------
For every evaluated event:
  - Annotated frame PNG showing: NVZ line, foot point/bbox, signed distance, label
  - Per-event row in foot_fault_events.csv
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.court_model import CourtGeometryModel
from src.court_registration import LineModel
from src.foot_localizer import localize_foot, localize_foot_event

logger = logging.getLogger(__name__)

# Label colours (BGR)
_COLOR_LEGAL     = (0, 220, 60)
_COLOR_FAULT     = (0, 60, 220)
_COLOR_UNCERTAIN = (0, 200, 220)
_COLOR_FOOT      = (255, 100, 0)
_COLOR_LINE      = (0, 220, 50)
_COLOR_LEFT_FOOT = (255, 120, 0)
_COLOR_RIGHT_FOOT = (180, 0, 255)
_COLOR_PERSON = (80, 220, 255)
_COLOR_LOWER_BODY = (255, 255, 0)


# ── registration loader ───────────────────────────────────────────────────────

def load_registration_csv(path: Path) -> dict[int, dict]:
    """Return dict of frame_index → row dict from per_frame_transforms.csv."""
    rows: dict[int, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row["frame_index"])
            rows[fi] = row
    return rows


def _model_from_reg_row(row: dict) -> Optional[CourtGeometryModel]:
    """Reconstruct CourtGeometryModel from a registration CSV row."""
    try:
        anchors = {
            "kitchen_near_left":  [float(row["kitchen_near_p1_x"]), float(row["kitchen_near_p1_y"])],
            "kitchen_near_right": [float(row["kitchen_near_p2_x"]), float(row["kitchen_near_p2_y"])],
        }
        if row.get("kitchen_far_p1_x") not in (None, "", "None"):
            anchors["kitchen_far_left"]  = [float(row["kitchen_far_p1_x"]), float(row["kitchen_far_p1_y"])]
            anchors["kitchen_far_right"] = [float(row["kitchen_far_p2_x"]), float(row["kitchen_far_p2_y"])]
        return CourtGeometryModel(anchors)
    except Exception as e:
        logger.warning(f"Could not build court model from row: {e}")
        return None


def _model_from_manual_override(override: dict) -> Optional[CourtGeometryModel]:
    """Build a CourtGeometryModel from a manual line override dict."""
    try:
        anchors = {
            "kitchen_near_left":  override["kitchen_near_left"],
            "kitchen_near_right": override["kitchen_near_right"],
        }
        if "kitchen_far_left" in override:
            anchors["kitchen_far_left"]  = override["kitchen_far_left"]
            anchors["kitchen_far_right"] = override["kitchen_far_right"]
        return CourtGeometryModel(anchors)
    except Exception as e:
        logger.warning(f"Could not build court model from manual override: {e}")
        return None


def load_manual_line_overrides(path: Path) -> dict:
    """
    Load clip-level NVZ line override file.

    JSON format:
    {
      "kitchen_near_left":  [x, y],
      "kitchen_near_right": [x, y],
      "kitchen_far_left":   [x, y],   # optional
      "kitchen_far_right":  [x, y]    # optional
    }
    """
    with open(path) as f:
        return json.load(f)


# ── NVZ boundary selection ────────────────────────────────────────────────────

def _select_boundary(
    model: CourtGeometryModel,
    side: str,
) -> Optional[LineModel]:
    """Return the appropriate NVZ boundary line for the given side ('left'/'right'/'near')."""
    if side == "left":
        return model.left_boundary_line
    elif side == "right":
        return model.right_boundary_line
    else:
        return model.near_kitchen_line


def _court_center_x(model: CourtGeometryModel) -> float:
    xs = [
        float(model.near_kitchen_line.p1[0]),
        float(model.near_kitchen_line.p2[0]),
    ]
    if model.far_kitchen_line is not None:
        xs.extend([
            float(model.far_kitchen_line.p1[0]),
            float(model.far_kitchen_line.p2[0]),
        ])
    return float(sum(xs) / len(xs))


def infer_active_side(
    event: dict,
    model: Optional[CourtGeometryModel],
    default_side: str = "left",
) -> dict:
    """Infer which player's side is active for this event."""
    override_side = event.get("override_active_side") or event.get("active_side_override")
    if override_side in {"left", "right"}:
        return {
            "active_side": override_side,
            "inferred_active_side": override_side,
            "active_side_source": "review_override",
            "side_confidence": 1.0,
            "court_center_x": _court_center_x(model) if model is not None else None,
        }

    if model is None:
        return {
            "active_side": default_side,
            "inferred_active_side": default_side,
            "active_side_source": "config_default",
            "side_confidence": 0.0,
            "court_center_x": None,
            "ball_support_n": 0,
        }

    center_x = _court_center_x(model)

    ball_window = event.get("ball_window") or []
    min_conf = float(event.get("active_side_min_ball_confidence", 0.25))
    if ball_window:
        weighted_left = 0.0
        weighted_right = 0.0
        weighted_offsets = []
        valid_n = 0
        event_fi = int(event.get("frame_index", 0))
        sigma = max(1.0, float(event.get("active_side_temporal_sigma_frames", 6.0)))

        for row in ball_window:
            bx = row.get("ball_x")
            if bx is None:
                continue
            conf = float(row.get("confidence") or 0.0)
            if conf < min_conf:
                continue
            dt = abs(int(row.get("frame_index", event_fi)) - event_fi)
            temporal_w = float(np.exp(-0.5 * (dt / sigma) ** 2))
            weight = max(0.05, conf) * temporal_w
            offset = float(bx) - center_x
            weighted_offsets.append(offset * weight)
            if offset <= 0:
                weighted_left += weight
            else:
                weighted_right += weight
            valid_n += 1

        if valid_n > 0:
            active = "left" if weighted_left >= weighted_right else "right"
            total_w = max(1e-6, weighted_left + weighted_right)
            conf = abs(weighted_left - weighted_right) / total_w
            mean_offset = sum(weighted_offsets) / total_w
            return {
                "active_side": active,
                "inferred_active_side": active,
                "active_side_source": "ball_window_vote",
                "side_confidence": round(float(min(1.0, 0.35 + conf)), 3),
                "court_center_x": round(center_x, 2),
                "ball_support_n": valid_n,
                "ball_mean_offset_x": round(float(mean_offset), 2),
            }

    ball_x = event.get("ball_x")
    if ball_x is not None:
        active = "left" if float(ball_x) <= center_x else "right"
        dist = abs(float(ball_x) - center_x)
        conf = min(1.0, dist / 250.0)
        return {
            "active_side": active,
            "inferred_active_side": active,
            "active_side_source": "ball_x",
            "side_confidence": round(conf, 3),
            "court_center_x": round(center_x, 2),
            "ball_support_n": 1,
        }

    return {
        "active_side": default_side,
        "inferred_active_side": default_side,
        "active_side_source": "config_default",
        "side_confidence": 0.0,
        "court_center_x": round(center_x, 2),
        "ball_support_n": 0,
    }


# ── decision ──────────────────────────────────────────────────────────────────

def _classify_distance(
    signed_dist: float,
    fault_threshold_px: float,
    uncertain_margin_px: float,
) -> str:
    """
    Sign convention from LineModel.signed_distance:
    The legal side of the NVZ boundary has POSITIVE signed distance
    (the foot is behind the line = legal).

    Negative signed distance → foot crossed into kitchen = fault.
    """
    if signed_dist > uncertain_margin_px:
        return "legal_volley"
    elif signed_dist < -fault_threshold_px:
        return "foot_fault_volley"
    else:
        return "uncertain"


def _manual_foot_result_from_event(event: dict) -> Optional[dict]:
    ox = event.get("override_foot_x")
    oy = event.get("override_foot_y")
    if ox is None or oy is None:
        return None
    return {
        "foot_x": float(ox),
        "foot_y": float(oy),
        "confidence": 1.0,
        "mode": "manual_point",
        "bbox": None,
        "roi_bbox": None,
        "low_confidence": False,
        "temporal_support_n": 1,
    }


def _evaluate_side(
    side: str,
    event: dict,
    frame: np.ndarray,
    frames: list[np.ndarray],
    frame_indices: list[int],
    frame_index: int,
    model: CourtGeometryModel,
    foot_cfg: dict,
    fault_threshold: float,
    uncertain_margin: float,
    review_conf_threshold: float,
) -> dict:
    boundary = _select_boundary(model, side)
    if boundary is None:
        return {
            "side": side,
            "boundary": None,
            "foot_result": None,
            "signed_dist_px": None,
            "label": "uncertain",
            "review_required": True,
        }

    manual_foot = None
    active_override_side = event.get("override_active_side") or event.get("active_side_override")
    if active_override_side == side:
        manual_foot = _manual_foot_result_from_event(event)

    if manual_foot is not None:
        foot_result = manual_foot
    elif str(foot_cfg.get("mode", "background_subtraction")) == "event_hybrid":
        foot_result = localize_foot_event(
            frames=frames,
            frame_indices=frame_indices,
            target_index=frame_index,
            boundary=boundary,
            cfg=foot_cfg,
        )
    else:
        foot_result = localize_foot(frame, frame_index=frame_index, cfg=foot_cfg)

    if foot_result is None:
        return {
            "side": side,
            "boundary": boundary,
            "foot_result": None,
            "signed_dist_px": None,
            "label": "uncertain",
            "review_required": True,
        }

    foot_pt = (float(foot_result["foot_x"]), float(foot_result["foot_y"]))
    signed_dist = float(boundary.signed_distance(foot_pt))
    label = _classify_distance(signed_dist, fault_threshold, uncertain_margin)
    review_required = bool(
        foot_result.get("low_confidence") or
        float(foot_result.get("confidence", 0.0)) < review_conf_threshold
    )
    if review_required:
        label = "uncertain"

    return {
        "side": side,
        "boundary": boundary,
        "foot_result": foot_result,
        "signed_dist_px": signed_dist,
        "label": label,
        "review_required": review_required,
    }


def analyze_event_feet(
    event: dict,
    frame: np.ndarray,
    frames: list[np.ndarray],
    frame_indices: list[int],
    frame_index: int,
    model: Optional[CourtGeometryModel],
    cfg: dict,
) -> dict:
    fault_threshold = float(cfg.get("fault_threshold_px", 5.0))
    uncertain_margin = float(cfg.get("uncertain_margin_px", 15.0))
    default_side = str(cfg.get("nvz_side", "left"))
    foot_cfg = cfg.get("foot_localizer", {"mode": "background_subtraction"})
    review_conf_threshold = float(foot_cfg.get("low_confidence_threshold", 0.45))
    min_side_confidence = float(cfg.get("active_side_min_confidence", 0.7))
    min_ball_support = int(cfg.get("active_side_min_support_n", 2))

    if model is None:
        side_info = infer_active_side(event, model, default_side=default_side)
        return {
            "active_side": side_info["active_side"],
            "inferred_active_side": side_info["inferred_active_side"],
            "active_side_source": side_info["active_side_source"],
            "side_confidence": side_info["side_confidence"],
            "court_center_x": side_info["court_center_x"],
            "ball_support_n": side_info.get("ball_support_n", 0),
            "side_results": {},
            "label": "uncertain",
            "signed_dist_px": None,
            "foot_result": None,
            "review_required": True,
            "ball_x": event.get("ball_x"),
            "ball_y": event.get("ball_y"),
        }

    side_info = infer_active_side(event, model, default_side=default_side)
    active_side = side_info["active_side"]
    side_results = {}
    for side in ("left", "right"):
        side_results[side] = _evaluate_side(
            side=side,
            event=event,
            frame=frame,
            frames=frames,
            frame_indices=frame_indices,
            frame_index=frame_index,
            model=model,
            foot_cfg=foot_cfg,
            fault_threshold=fault_threshold,
            uncertain_margin=uncertain_margin,
            review_conf_threshold=review_conf_threshold,
        )

    primary = side_results.get(active_side) or side_results.get(default_side) or next(iter(side_results.values()))
    label = primary["label"]
    review_required = bool(primary["review_required"])
    side_confidence = float(side_info.get("side_confidence", 0.0) or 0.0)
    ball_support_n = int(side_info.get("ball_support_n", 0) or 0)
    if side_info["active_side_source"] == "config_default":
        review_required = True
        label = "uncertain"
    elif side_info["active_side_source"] != "review_override":
        if side_confidence < min_side_confidence or ball_support_n < min_ball_support:
            review_required = True
            label = "uncertain"

    return {
        "active_side": active_side,
        "inferred_active_side": side_info["inferred_active_side"],
        "active_side_source": side_info["active_side_source"],
        "side_confidence": side_confidence,
        "court_center_x": side_info["court_center_x"],
        "ball_support_n": ball_support_n,
        "side_results": side_results,
        "label": label,
        "signed_dist_px": primary["signed_dist_px"],
        "foot_result": primary["foot_result"],
        "review_required": review_required,
        "ball_x": event.get("ball_x"),
        "ball_y": event.get("ball_y"),
    }


# ── verification frame renderer ───────────────────────────────────────────────

def _annotate_event_frame(
    frame: np.ndarray,
    model: CourtGeometryModel,
    analysis: dict,
    label: str,
    frame_index: int,
    timestamp_s: float,
    side: str,
) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]

    # draw court model (NVZ lines)
    for line in (model.left_boundary_line, model.right_boundary_line, model.near_kitchen_line):
        if line is None:
            continue
        pt1, pt2 = line.endpoints_in_frame(W, H)
        cv2.line(out, pt1, pt2, _COLOR_LINE, 2)

    # highlight the relevant boundary
    active_side = analysis.get("active_side", side)
    ball_x = analysis.get("ball_x")
    ball_y = analysis.get("ball_y")

    for side_name, color in (("left", _COLOR_LEFT_FOOT), ("right", _COLOR_RIGHT_FOOT)):
        side_result = analysis.get("side_results", {}).get(side_name, {})
        boundary = side_result.get("boundary")
        if boundary is not None:
            pt1, pt2 = boundary.endpoints_in_frame(W, H)
            line_color = (0, 255, 255) if side_name == active_side else _COLOR_LINE
            line_thickness = 3 if side_name == active_side else 2
            cv2.line(out, pt1, pt2, line_color, line_thickness)

        foot_result = side_result.get("foot_result")
        if foot_result is None:
            continue
        fx = int(round(float(foot_result["foot_x"])))
        fy = int(round(float(foot_result["foot_y"])))
        radius = 10 if side_name == active_side else 7
        cv2.circle(out, (fx, fy), radius, color, -1)
        cv2.circle(out, (fx, fy), radius, (255, 255, 255), 2 if side_name == active_side else 1)
        bbox = foot_result.get("bbox")
        if bbox is not None:
            bx, by, bw, bh = bbox
            cv2.rectangle(out, (int(bx), int(by)), (int(bx + bw), int(by + bh)), color, 1)
        person_bbox = foot_result.get("person_bbox")
        if person_bbox is not None:
            px, py, pw, ph = [int(v) for v in person_bbox]
            cv2.rectangle(out, (px, py), (px + pw, py + ph), _COLOR_PERSON, 2 if side_name == active_side else 1)
        pose_keypoints = foot_result.get("pose_keypoints")
        if pose_keypoints is not None:
            leg_color = color
            for kp_idx in (11, 12, 13, 14, 15, 16):
                kp = pose_keypoints[kp_idx]
                if float(kp[2]) < 0.35:
                    continue
                cv2.circle(out, (int(round(float(kp[0]))), int(round(float(kp[1])))), 4, leg_color, -1)
        lower_body_bbox = foot_result.get("lower_body_bbox")
        if lower_body_bbox is not None:
            lx0, ly0, lx1, ly1 = [int(v) for v in lower_body_bbox]
            cv2.rectangle(out, (lx0, ly0), (lx1, ly1), _COLOR_LOWER_BODY, 2 if side_name == active_side else 1)
        roi_bbox = foot_result.get("roi_bbox")
        if roi_bbox is not None and side_name == active_side:
            x0, y0, x1, y1 = [int(v) for v in roi_bbox]
            cv2.rectangle(out, (x0, y0), (x1, y1), (180, 180, 255), 2)

    if ball_x is not None and ball_y is not None:
        bx = int(round(float(ball_x)))
        by = int(round(float(ball_y)))
        cv2.circle(out, (bx, by), 9, (0, 255, 255), 2)
        cv2.circle(out, (bx, by), 3, (0, 255, 255), -1)

    # label colour
    if label == "legal_volley":
        color = _COLOR_LEGAL
    elif label == "foot_fault_volley":
        color = _COLOR_FAULT
    else:
        color = _COLOR_UNCERTAIN

    # overlay text
    primary_foot = analysis.get("foot_result")
    signed_dist = analysis.get("signed_dist_px")
    dist_str = f"{signed_dist:+.1f}px" if signed_dist is not None else "N/A"
    foot_conf = f"{primary_foot['confidence']:.2f}" if primary_foot else "N/A"
    left_dist = analysis.get("side_results", {}).get("left", {}).get("signed_dist_px")
    right_dist = analysis.get("side_results", {}).get("right", {}).get("signed_dist_px")
    left_str = f"{left_dist:+.1f}" if left_dist is not None else "N/A"
    right_str = f"{right_dist:+.1f}" if right_dist is not None else "N/A"
    line1 = (
        f"f={frame_index}  t={timestamp_s:.2f}s  active={active_side} "
        f"({analysis.get('active_side_source', 'unknown')}, conf={analysis.get('side_confidence', 0.0):.2f}, "
        f"n={analysis.get('ball_support_n', 0)})"
    )
    line2 = f"active_dist={dist_str}  foot_conf={foot_conf}  ball=({ball_x:.0f},{ball_y:.0f})" if ball_x is not None and ball_y is not None else f"active_dist={dist_str}  foot_conf={foot_conf}"
    line3 = f"left_dist={left_str}px  right_dist={right_str}px"
    detector_src = primary_foot.get("person_detector_source") if primary_foot else None
    line4 = f"detector={detector_src or 'n/a'}  cyan=person  yellow=lower-body"
    line5 = f"LABEL: {label.upper()}"
    if primary_foot and primary_foot.get("low_confidence"):
        line5 += "  [REVIEW]"

    for i, text in enumerate([line1, line2, line3, line4, line5]):
        y = 26 + i * 22
        c = color if i == 4 else (255, 255, 255)
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)

    return out


def _read_frame_window(
    cap: cv2.VideoCapture,
    center_frame: int,
    radius: int,
) -> tuple[list[np.ndarray], list[int]]:
    frames: list[np.ndarray] = []
    indices: list[int] = []
    for fi in range(max(0, center_frame - radius), center_frame + radius + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
        indices.append(fi)
    return frames, indices


def _build_zoom_panel(
    frame: np.ndarray,
    foot_result: Optional[dict],
    boundary: Optional[LineModel],
    title: str = "Foot ROI",
    panel_size: tuple[int, int] = (360, 360),
) -> np.ndarray:
    H, W = frame.shape[:2]
    if foot_result is not None and foot_result.get("roi_bbox") is not None:
        x0, y0, x1, y1 = [int(v) for v in foot_result["roi_bbox"]]
    elif foot_result is not None:
        fx = int(round(float(foot_result["foot_x"])))
        fy = int(round(float(foot_result["foot_y"])))
        pad = 120
        x0, y0, x1, y1 = fx - pad, fy - pad, fx + pad, fy + pad
    else:
        x0, y0, x1, y1 = 0, max(0, H - 300), min(W, 360), H

    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W))
    y1 = max(y0 + 1, min(y1, H))

    crop = frame[y0:y1, x0:x1].copy()
    if crop.size == 0:
        crop = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
    else:
        if boundary is not None:
            pt1, pt2 = boundary.endpoints_in_frame(W, H)
            cv2.line(crop, (pt1[0] - x0, pt1[1] - y0), (pt2[0] - x0, pt2[1] - y0), (0, 255, 255), 2)
        if foot_result is not None:
            fx = int(round(float(foot_result["foot_x"]))) - x0
            fy = int(round(float(foot_result["foot_y"]))) - y0
            cv2.circle(crop, (fx, fy), 10, _COLOR_FOOT, -1)
            cv2.circle(crop, (fx, fy), 10, (255, 255, 255), 2)
            bbox = foot_result.get("bbox")
            if bbox is not None:
                bx, by, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(crop, (bx - x0, by - y0), (bx + bw - x0, by + bh - y0), _COLOR_FOOT, 2)
            person_bbox = foot_result.get("person_bbox")
            if person_bbox is not None:
                px, py, pw, ph = [int(v) for v in person_bbox]
                cv2.rectangle(crop, (px - x0, py - y0), (px + pw - x0, py + ph - y0), _COLOR_PERSON, 2)
            pose_keypoints = foot_result.get("pose_keypoints")
            if pose_keypoints is not None:
                for kp_idx in (11, 12, 13, 14, 15, 16):
                    kp = pose_keypoints[kp_idx]
                    if float(kp[2]) < 0.35:
                        continue
                    cv2.circle(crop, (int(round(float(kp[0]))) - x0, int(round(float(kp[1]))) - y0), 4, _COLOR_FOOT, -1)
            lower_body_bbox = foot_result.get("lower_body_bbox")
            if lower_body_bbox is not None:
                lx0, ly0, lx1, ly1 = [int(v) for v in lower_body_bbox]
                cv2.rectangle(crop, (lx0 - x0, ly0 - y0), (lx1 - x0, ly1 - y0), _COLOR_LOWER_BODY, 2)
        crop = cv2.resize(crop, panel_size, interpolation=cv2.INTER_LINEAR)

    cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (220, 220, 220), 2)
    cv2.putText(crop, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(crop, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(crop, "cyan=person  yellow=lower-body  dot=contact", (10, crop.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 3)
    cv2.putText(crop, "cyan=person  yellow=lower-body  dot=contact", (10, crop.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
    return crop


# ── main entry point ──────────────────────────────────────────────────────────

def run_foot_fault_pipeline(
    volley_events: list[dict],
    video_path: Path,
    output_dir: Path,
    cfg: dict,
    registration_csv: Optional[Path] = None,
    manual_line_override_path: Optional[Path] = None,
    ref_annotations_path: Optional[Path] = None,
) -> list[dict]:
    """
    Evaluate foot fault for each volley event.

    Parameters
    ----------
    volley_events           : list of event dicts with at least 'frame_index'
    video_path              : source video
    output_dir              : where to write results
    cfg                     : foot_fault config section (thresholds, foot_localizer, side)
    registration_csv        : path to per_frame_transforms.csv from court_reg_v3
    manual_line_override_path : optional JSON with manual line overrides
    ref_annotations_path    : fallback — load ref model from annotations_v3.json

    Returns list of per-event result dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "event_frames"
    frames_dir.mkdir(exist_ok=True)
    roi_dir = output_dir / "roi_frames"
    roi_dir.mkdir(exist_ok=True)

    foot_cfg = cfg.get("foot_localizer", {"mode": "background_subtraction"})
    temporal_radius = int(foot_cfg.get("temporal_window_radius", 1))

    # ── load registration ─────────────────────────────────────────────────────
    reg_rows: dict[int, dict] = {}
    if registration_csv and registration_csv.exists():
        reg_rows = load_registration_csv(registration_csv)
        logger.info(f"Loaded registration CSV: {len(reg_rows)} rows from {registration_csv}")

    manual_override_model: Optional[CourtGeometryModel] = None
    if manual_line_override_path and manual_line_override_path.exists():
        override_data = load_manual_line_overrides(manual_line_override_path)
        manual_override_model = _model_from_manual_override(override_data)
        logger.info(f"Manual line override loaded from {manual_line_override_path}")

    # ── fallback: load ref model from annotations ─────────────────────────────
    ref_model: Optional[CourtGeometryModel] = None
    if ref_annotations_path and ref_annotations_path.exists():
        import json as _json
        with open(ref_annotations_path) as f:
            ann = _json.load(f)
        frames = ann.get("annotated_frames", [])
        if frames:
            anchors = frames[0].get("anchors", {})
            try:
                ref_model = CourtGeometryModel(anchors)
            except Exception:
                pass

    def _get_model(frame_index: int) -> Optional[CourtGeometryModel]:
        if manual_override_model is not None:
            return manual_override_model
        if frame_index in reg_rows:
            return _model_from_reg_row(reg_rows[frame_index])
        # nearest registered frame
        if reg_rows:
            nearest = min(reg_rows.keys(), key=lambda k: abs(k - frame_index))
            return _model_from_reg_row(reg_rows[nearest])
        return ref_model

    # ── open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    results = []
    for event in volley_events:
        fi = int(event["frame_index"])
        ts = float(event.get("timestamp_s", fi / src_fps))
        frames, frame_indices = _read_frame_window(cap, fi, temporal_radius)
        if not frames:
            logger.warning(f"Could not read frame window around {fi}")
            continue
        try:
            target_pos = frame_indices.index(fi)
        except ValueError:
            target_pos = len(frames) // 2
        frame = frames[target_pos]

        model = _get_model(fi)
        if model is None:
            label = "uncertain"
            analysis = {
                "active_side": event.get("active_side") or cfg.get("nvz_side", "left"),
                "inferred_active_side": event.get("active_side") or cfg.get("nvz_side", "left"),
                "active_side_source": "config_default",
                "side_confidence": 0.0,
                "court_center_x": None,
                "side_results": {},
                "label": label,
                "signed_dist_px": None,
                "foot_result": None,
                "review_required": True,
                "ball_x": event.get("ball_x"),
                "ball_y": event.get("ball_y"),
            }
        else:
            analysis = analyze_event_feet(
                event=event,
                frame=frame,
                frames=frames,
                frame_indices=frame_indices,
                frame_index=fi,
                model=model,
                cfg=cfg,
            )
            label = analysis["label"]

        # write annotated frame
        if model is not None:
            ann_frame = _annotate_event_frame(
                frame, model, analysis, label, fi, ts, analysis["active_side"]
            )
        else:
            ann_frame = frame.copy()
            cv2.putText(ann_frame, f"f={fi} t={ts:.2f}s  NO REGISTRATION  [{label}]",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        primary_side = analysis.get("active_side")
        primary_side_result = analysis.get("side_results", {}).get(primary_side, {})
        primary_boundary = primary_side_result.get("boundary")
        primary_foot = primary_side_result.get("foot_result")
        roi_panel = _build_zoom_panel(
            frame,
            primary_foot,
            primary_boundary,
            title=f"{primary_side.title()} ROI" if primary_side else "Foot ROI",
        )
        roi_path = roi_dir / f"event_{fi:05d}_roi.png"
        cv2.imwrite(str(roi_path), roi_panel)

        frame_path = frames_dir / f"event_{fi:05d}.png"
        cv2.imwrite(str(frame_path), ann_frame)

        row = {
            "frame_index": fi,
            "timestamp_s": round(ts, 4),
            "side": analysis.get("active_side"),
            "active_side": analysis.get("active_side"),
            "inferred_active_side": analysis.get("inferred_active_side"),
            "active_side_source": analysis.get("active_side_source"),
            "active_side_confidence": analysis.get("side_confidence"),
            "label": label,
            "ball_x": round(float(analysis["ball_x"]), 2) if analysis.get("ball_x") is not None else None,
            "ball_y": round(float(analysis["ball_y"]), 2) if analysis.get("ball_y") is not None else None,
            "signed_dist_px": round(analysis["signed_dist_px"], 2) if analysis.get("signed_dist_px") is not None else None,
            "foot_x": round(primary_foot["foot_x"], 2) if primary_foot else None,
            "foot_y": round(primary_foot["foot_y"], 2) if primary_foot else None,
            "foot_confidence": round(primary_foot["confidence"], 3) if primary_foot else None,
            "foot_mode": primary_foot["mode"] if primary_foot else None,
            "left_signed_dist_px": round(analysis.get("side_results", {}).get("left", {}).get("signed_dist_px"), 2) if analysis.get("side_results", {}).get("left", {}).get("signed_dist_px") is not None else None,
            "right_signed_dist_px": round(analysis.get("side_results", {}).get("right", {}).get("signed_dist_px"), 2) if analysis.get("side_results", {}).get("right", {}).get("signed_dist_px") is not None else None,
            "left_foot_x": round(analysis.get("side_results", {}).get("left", {}).get("foot_result", {}).get("foot_x"), 2) if analysis.get("side_results", {}).get("left", {}).get("foot_result") else None,
            "left_foot_y": round(analysis.get("side_results", {}).get("left", {}).get("foot_result", {}).get("foot_y"), 2) if analysis.get("side_results", {}).get("left", {}).get("foot_result") else None,
            "right_foot_x": round(analysis.get("side_results", {}).get("right", {}).get("foot_result", {}).get("foot_x"), 2) if analysis.get("side_results", {}).get("right", {}).get("foot_result") else None,
            "right_foot_y": round(analysis.get("side_results", {}).get("right", {}).get("foot_result", {}).get("foot_y"), 2) if analysis.get("side_results", {}).get("right", {}).get("foot_result") else None,
            "review_required": bool(analysis.get("review_required") or label == "uncertain"),
            "frame_path": str(frame_path),
            "roi_frame_path": str(roi_path),
        }
        results.append(row)
        logger.info(
            f"  frame {fi}: active_side={row['active_side']} label={label}  "
            f"dist={row['signed_dist_px']}px  foot_conf={row['foot_confidence']}"
        )

    cap.release()

    # write CSV
    csv_path = output_dir / "foot_fault_events.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    # summary
    from collections import Counter
    label_counts = Counter(r["label"] for r in results)
    summary = {
        "n_events": len(results),
        "label_counts": dict(label_counts),
        "fault_threshold_px": float(cfg.get("fault_threshold_px", 5.0)),
        "uncertain_margin_px": float(cfg.get("uncertain_margin_px", 15.0)),
        "default_nvz_side": str(cfg.get("nvz_side", "left")),
        "foot_mode": foot_cfg.get("mode"),
    }
    summary_path = output_dir / "summary.json"
    import json as _json2
    with open(summary_path, "w") as f:
        _json2.dump(summary, f, indent=2)

    logger.info(
        f"Foot-fault pipeline done: {len(results)} events  labels={dict(label_counts)}  "
        f"CSV={csv_path}"
    )
    return results
