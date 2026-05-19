"""
Foot localization — presentation-first implementation.

Supports four modes, selectable via cfg['mode']:

  background_subtraction
      MOG2 background model + morphological cleanup to isolate moving player
      regions. The foot point is the centroid of the lowest moving blob
      (by image-y) within an optional ROI.

  roi_threshold
      Simple HSV or grayscale thresholding inside a configurable ROI strip.
      Works on a single frame without needing a reference background.

  manual_point
      Load a pre-defined foot point from a JSON override file. Overrides
      are keyed by frame_index; if no exact match, the nearest override entry
      is used. Confidence is always 1.0 (human-annotated).

  event_hybrid
      Candidate-event localizer for real video. Uses a boundary-aware ROI,
      background subtraction cue, threshold cue, morphology cleanup, and
      short temporal smoothing across neighboring frames. Chooses the blob
      closest to the selected NVZ boundary rather than simply the largest or
      lowest blob.

Returns
-------
dict or None
    {
        "foot_x":      float,
        "foot_y":      float,            # contact point (bottom-most plausible point)
        "confidence":  float 0–1,
        "mode":        str,
        "bbox":        (x, y, w, h) | None,
        "roi_bbox":    (x0, y0, x1, y1) | None,
        "low_confidence": bool,
        "temporal_support_n": int,
        "debug": {...}  # optional masks / per-frame detections for review rendering
    }
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.court_registration import LineModel

logger = logging.getLogger(__name__)


# ── MOG2 background model (shared across calls when using bg_subtraction) ────

_bg_subtractor: Optional[cv2.BackgroundSubtractorMOG2] = None
_hog_person_detector: Optional[cv2.HOGDescriptor] = None
_pose_net: Optional[cv2.dnn.Net] = None


def reset_background_model() -> None:
    global _bg_subtractor
    _bg_subtractor = None


def _ensure_hog_person_detector() -> cv2.HOGDescriptor:
    global _hog_person_detector
    if _hog_person_detector is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _hog_person_detector = hog
    return _hog_person_detector


def _ensure_pose_net(cfg: dict) -> cv2.dnn.Net:
    global _pose_net
    if _pose_net is None:
        model_path = Path(str(cfg.get("pose_model_path", "models/yolov8n-pose.onnx")))
        if not model_path.exists():
            raise FileNotFoundError(f"Pose model not found: {model_path}")
        _pose_net = cv2.dnn.readNetFromONNX(str(model_path))
    return _pose_net


def _ensure_bg_subtractor(cfg: dict) -> cv2.BackgroundSubtractorMOG2:
    global _bg_subtractor
    if _bg_subtractor is None:
        _bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=int(cfg.get("bg_history", 200)),
            varThreshold=float(cfg.get("bg_var_threshold", 40.0)),
            detectShadows=False,
        )
    return _bg_subtractor


# ── ROI helpers ───────────────────────────────────────────────────────────────

def _roi_from_cfg(cfg: dict, frame_shape: tuple) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) ROI from config, clipped to frame."""
    H, W = frame_shape[:2]
    roi = cfg.get("roi", {})
    x0 = int(roi.get("x0", 0))
    y0 = int(roi.get("y0", 0))
    x1 = int(roi.get("x1", W))
    y1 = int(roi.get("y1", H))
    return (
        max(0, min(x0, W - 1)),
        max(0, min(y0, H - 1)),
        max(0, min(x1, W)),
        max(0, min(y1, H)),
    )


def _clip_roi(roi: tuple[int, int, int, int], frame_shape: tuple) -> tuple[int, int, int, int]:
    H, W = frame_shape[:2]
    x0, y0, x1, y1 = roi
    return (
        max(0, min(int(x0), W - 1)),
        max(0, min(int(y0), H - 1)),
        max(1, min(int(x1), W)),
        max(1, min(int(y1), H)),
    )


def _intersect_roi(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    union = float(aw * ah + bw * bh - inter)
    if union <= 1e-6:
        return 0.0
    return inter / union


def _non_max_suppress_boxes(boxes: list[dict], iou_thresh: float = 0.35) -> list[dict]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)
    kept: list[dict] = []
    for box in boxes:
        if all(_bbox_iou(box["bbox"], prev["bbox"]) < iou_thresh for prev in kept):
            kept.append(box)
    return kept


def _boundary_roi(
    boundary: Optional[LineModel],
    frame_shape: tuple,
    cfg: dict,
) -> tuple[int, int, int, int]:
    H, W = frame_shape[:2]
    if boundary is None:
        return _roi_from_cfg(cfg, frame_shape)

    pt1, pt2 = boundary.endpoints_in_frame(W, H)
    pad_x = int(cfg.get("boundary_pad_x", 140))
    pad_y = int(cfg.get("boundary_pad_y", 120))
    near_bottom_bonus = int(cfg.get("near_bottom_bonus_px", 80))
    roi = (
        min(pt1[0], pt2[0]) - pad_x,
        min(pt1[1], pt2[1]) - pad_y,
        max(pt1[0], pt2[0]) + pad_x,
        max(pt1[1], pt2[1]) + pad_y + near_bottom_bonus,
    )
    roi = _clip_roi(roi, frame_shape)

    cfg_roi = _roi_from_cfg(cfg, frame_shape)
    merged = _intersect_roi(roi, cfg_roi)
    if merged[2] <= merged[0] or merged[3] <= merged[1]:
        return roi
    return merged


def _detect_people_hog(
    frame: np.ndarray,
    search_roi: tuple[int, int, int, int],
    cfg: dict,
) -> list[dict]:
    x0, y0, x1, y1 = search_roi
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []

    hog = _ensure_hog_person_detector()
    scale_down = float(cfg.get("person_detector_scale_down", 0.6))
    detect_img = crop
    inv_scale = 1.0
    if 0.1 < scale_down < 1.0:
        w = max(64, int(crop.shape[1] * scale_down))
        h = max(128, int(crop.shape[0] * scale_down))
        detect_img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        inv_scale = 1.0 / scale_down

    rects, weights = hog.detectMultiScale(
        detect_img,
        winStride=(8, 8),
        padding=(8, 8),
        scale=float(cfg.get("person_detector_scale", 1.05)),
    )

    min_score = float(cfg.get("person_detector_min_score", 0.2))
    detections = []
    for rect, weight in zip(rects, weights):
        rx, ry, rw, rh = [int(v) for v in rect]
        score = float(weight)
        if score < min_score:
            continue
        bx = int(round(rx * inv_scale)) + x0
        by = int(round(ry * inv_scale)) + y0
        bw = int(round(rw * inv_scale))
        bh = int(round(rh * inv_scale))
        detections.append({
            "bbox": (bx, by, bw, bh),
            "score": score,
        })
    return _non_max_suppress_boxes(detections, iou_thresh=float(cfg.get("person_detector_nms_iou", 0.35)))


def _detect_people_pose(
    frame: np.ndarray,
    search_roi: tuple[int, int, int, int],
    cfg: dict,
) -> list[dict]:
    x0, y0, x1, y1 = search_roi
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []

    net = _ensure_pose_net(cfg)
    input_size = int(cfg.get("pose_input_size", 640))
    resized = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(
        resized,
        scalefactor=1.0 / 255.0,
        size=(input_size, input_size),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    raw = net.forward()
    if raw.ndim != 3:
        return []

    preds = raw[0].transpose(1, 0)
    conf_thresh = float(cfg.get("pose_confidence_threshold", 0.35))
    nms_thresh = float(cfg.get("pose_nms_threshold", 0.45))
    kpt_thresh = float(cfg.get("pose_keypoint_threshold", 0.35))
    sx = crop.shape[1] / float(input_size)
    sy = crop.shape[0] / float(input_size)

    boxes: list[list[int]] = []
    scores: list[float] = []
    decoded: list[dict] = []
    for row in preds:
        score = float(row[4])
        if score < conf_thresh:
            continue
        cx, cy, w, h = [float(v) for v in row[:4]]
        bx = cx - w * 0.5
        by = cy - h * 0.5
        full_x = bx * sx + x0
        full_y = by * sy + y0
        full_w = w * sx
        full_h = h * sy
        if full_w <= 2 or full_h <= 2:
            continue

        kpts = row[5:].reshape(17, 3).astype(np.float32)
        kpts[:, 0] = kpts[:, 0] * sx + x0
        kpts[:, 1] = kpts[:, 1] * sy + y0
        decoded.append({
            "bbox": (int(round(full_x)), int(round(full_y)), int(round(full_w)), int(round(full_h))),
            "score": score,
            "keypoints": kpts,
            "keypoint_threshold": kpt_thresh,
        })
        boxes.append([
            int(round(full_x)),
            int(round(full_y)),
            int(round(full_w)),
            int(round(full_h)),
        ])
        scores.append(score)

    if not boxes:
        return []

    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)
    if len(idxs) == 0:
        return []

    kept = []
    for idx in idxs.flatten().tolist():
        kept.append(decoded[idx])
    return kept


def _select_person_detection(
    detections: list[dict],
    boundary: Optional[LineModel],
    frame_shape: tuple,
    cfg: dict,
) -> Optional[dict]:
    if not detections:
        return None

    H, W = frame_shape[:2]
    best = None
    for det in detections:
        bx, by, bw, bh = det["bbox"]
        bottom_center = (bx + bw * 0.5, by + bh)
        area = float(bw * bh)
        dist = abs(float(boundary.signed_distance(bottom_center))) if boundary is not None else 0.0
        bottomness = min(1.0, bottom_center[1] / max(1.0, H))
        area_conf = min(1.0, area / float(cfg.get("expected_person_area", 90000.0)))
        boundary_sigma = float(cfg.get("person_boundary_sigma_px", 140.0))
        dist_score = float(np.exp(-dist / max(1.0, boundary_sigma)))
        score = 0.5 * dist_score + 0.25 * bottomness + 0.15 * area_conf + 0.10 * min(1.0, det["score"])
        det = dict(det)
        det["selection_score"] = score
        det["boundary_dist"] = dist
        if best is None or score > best["selection_score"]:
            best = det
    return best


def _select_pose_detection(
    detections: list[dict],
    boundary: Optional[LineModel],
    frame_shape: tuple,
    cfg: dict,
) -> Optional[dict]:
    if not detections:
        return None

    H, W = frame_shape[:2]
    best = None
    kpt_thresh = float(cfg.get("pose_keypoint_threshold", 0.35))
    for det in detections:
        bx, by, bw, bh = det["bbox"]
        kpts = det["keypoints"]
        visible = [kp for kp in kpts if float(kp[2]) >= kpt_thresh]
        if visible:
            bottom_y = max(float(kp[1]) for kp in visible)
            center_x = float(np.mean([kp[0] for kp in visible]))
        else:
            center_x = bx + bw * 0.5
            bottom_y = by + bh
        bottom_center = (center_x, bottom_y)
        area = float(max(1, bw * bh))
        dist = abs(float(boundary.signed_distance(bottom_center))) if boundary is not None else 0.0
        bottomness = min(1.0, bottom_y / max(1.0, H))
        area_conf = min(1.0, area / float(cfg.get("expected_person_area", 90000.0)))
        boundary_sigma = float(cfg.get("person_boundary_sigma_px", 140.0))
        dist_score = float(np.exp(-dist / max(1.0, boundary_sigma)))
        vis_conf = min(1.0, len(visible) / 6.0)
        score = 0.42 * dist_score + 0.24 * bottomness + 0.20 * min(1.0, det["score"]) + 0.14 * vis_conf
        det2 = dict(det)
        det2["selection_score"] = score
        det2["boundary_dist"] = dist
        if best is None or score > best["selection_score"]:
            best = det2
    return best


_COCO_KPT = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def _pose_leg_visibility(kpts: np.ndarray, side: str, conf_thresh: float) -> list[np.ndarray]:
    names = [f"{side}_hip", f"{side}_knee", f"{side}_ankle"]
    pts = []
    for name in names:
        kp = kpts[_COCO_KPT[name]]
        if float(kp[2]) >= conf_thresh:
            pts.append(kp)
    return pts


def _select_boundary_side_leg(
    pose_det: dict,
    boundary: Optional[LineModel],
    cfg: dict,
) -> tuple[str, list[np.ndarray]]:
    kpts = pose_det["keypoints"]
    conf_thresh = float(cfg.get("pose_keypoint_threshold", 0.35))
    left_pts = _pose_leg_visibility(kpts, "left", conf_thresh)
    right_pts = _pose_leg_visibility(kpts, "right", conf_thresh)

    def _leg_dist(pts: list[np.ndarray]) -> float:
        if not pts or boundary is None:
            return 9999.0
        return min(abs(float(boundary.signed_distance((float(p[0]), float(p[1]))))) for p in pts)

    left_dist = _leg_dist(left_pts)
    right_dist = _leg_dist(right_pts)

    if left_pts and (not right_pts or left_dist <= right_dist):
        return "left", left_pts
    if right_pts:
        return "right", right_pts

    # fallback to bbox geometry if keypoints are missing
    bx, by, bw, bh = pose_det["bbox"]
    if boundary is None:
        return "left", []
    left_pt = (bx + bw * 0.3, by + bh)
    right_pt = (bx + bw * 0.7, by + bh)
    if abs(float(boundary.signed_distance(left_pt))) <= abs(float(boundary.signed_distance(right_pt))):
        return "left", []
    return "right", []


def _leg_roi_from_pose(
    pose_det: dict,
    leg_side: str,
    leg_points: list[np.ndarray],
    frame_shape: tuple,
    cfg: dict,
) -> tuple[int, int, int, int]:
    bx, by, bw, bh = pose_det["bbox"]
    if leg_points:
        xs = [float(p[0]) for p in leg_points]
        ys = [float(p[1]) for p in leg_points]
        x_pad = float(cfg.get("pose_leg_roi_pad_x", 34.0))
        y_pad_top = float(cfg.get("pose_leg_roi_pad_top", 26.0))
        y_pad_bottom = float(cfg.get("pose_leg_roi_pad_bottom", 52.0))
        roi = (
            int(min(xs) - x_pad),
            int(min(ys) - y_pad_top),
            int(max(xs) + x_pad),
            int(max(ys) + y_pad_bottom),
        )
        return _clip_roi(roi, frame_shape)

    focus_frac = float(cfg.get("boundary_leg_focus_frac", 0.45))
    if leg_side == "left":
        x0 = bx
        x1 = bx + int(round(bw * focus_frac))
    else:
        x0 = bx + bw - int(round(bw * focus_frac))
        x1 = bx + bw
    top = by + int(round(bh * float(cfg.get("lower_body_start_frac", 0.52))))
    bottom = by + bh
    return _clip_roi((x0, top, x1, bottom), frame_shape)


def _pose_seed_point(
    pose_det: dict,
    leg_side: str,
    conf_thresh: float,
) -> tuple[float, float]:
    kpts = pose_det["keypoints"]
    ankle = kpts[_COCO_KPT[f"{leg_side}_ankle"]]
    knee = kpts[_COCO_KPT[f"{leg_side}_knee"]]
    hip = kpts[_COCO_KPT[f"{leg_side}_hip"]]
    if float(ankle[2]) >= conf_thresh:
        return float(ankle[0]), float(ankle[1])
    if float(knee[2]) >= conf_thresh and float(hip[2]) >= conf_thresh:
        dx = float(knee[0] - hip[0])
        dy = float(knee[1] - hip[1])
        return float(knee[0] + 0.65 * dx), float(knee[1] + 0.75 * dy)
    if float(knee[2]) >= conf_thresh:
        return float(knee[0]), float(knee[1] + 60.0)
    return float(hip[0]), float(hip[1] + 120.0)


def _boundary_side_lower_body_bbox(
    person_bbox: tuple[int, int, int, int],
    boundary: Optional[LineModel],
    frame_shape: tuple,
    cfg: dict,
) -> tuple[int, int, int, int]:
    px, py, pw, ph = person_bbox
    if pw <= 0 or ph <= 0:
        return person_bbox

    lower_start_frac = float(cfg.get("lower_body_start_frac", 0.52))
    top = py + int(round(ph * lower_start_frac))
    bottom = py + ph

    left_x = px
    right_x = px + pw
    if boundary is not None:
        left_corner = (left_x, bottom)
        right_corner = (right_x, bottom)
        left_dist = abs(float(boundary.signed_distance(left_corner)))
        right_dist = abs(float(boundary.signed_distance(right_corner)))
        focus_frac = float(cfg.get("boundary_leg_focus_frac", 0.58))
        half_w = max(24, int(round(pw * focus_frac)))
        if left_dist <= right_dist:
            x0 = left_x
            x1 = left_x + half_w
        else:
            x0 = right_x - half_w
            x1 = right_x
    else:
        inset = int(round(pw * 0.18))
        x0 = left_x + inset
        x1 = right_x - inset

    return _clip_roi((x0, top, x1, bottom), frame_shape)


def _bottom_blob(mask: np.ndarray, min_area: float) -> Optional[dict]:
    """Return the lowest (max image-y) centroid + bbox of valid blobs."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] < 1e-6:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)
        foot_y = float(y + h)  # bottom of bounding box
        if best is None or foot_y > best["foot_y"]:
            best = {
                "cx": cx,
                "cy": cy,
                "foot_y": foot_y,
                "bbox": (x, y, w, h),
                "area": area,
            }
    return best


def _manual_override_entry(frame_index: int, cfg: dict) -> Optional[dict]:
    overrides = _manual_overrides
    if not overrides:
        override_path = cfg.get("override_file")
        if override_path:
            load_manual_overrides(Path(override_path))
            overrides = _manual_overrides
    if not overrides:
        return None
    return min(overrides, key=lambda e: abs(int(e["frame_index"]) - frame_index))


def _manual_result(frame_index: int, cfg: dict) -> Optional[dict]:
    best = _manual_override_entry(frame_index, cfg)
    if best is None:
        return None
    return {
        "foot_x": float(best["foot_x"]),
        "foot_y": float(best["foot_y"]),
        "confidence": 1.0,
        "mode": "manual_point",
        "bbox": None,
        "roi_bbox": None,
        "low_confidence": False,
        "temporal_support_n": 1,
    }


def _threshold_mask(roi: np.ndarray, cfg: dict) -> np.ndarray:
    threshold_mode = str(cfg.get("threshold_mode", "value"))
    if threshold_mode == "hsv":
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lo = np.array(cfg.get("hsv_lower", [0, 0, 0]), dtype=np.uint8)
        hi = np.array(cfg.get("hsv_upper", [180, 255, 80]), dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = int(cfg.get("gray_threshold", 80))
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return mask


def _cleanup_mask(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _contact_point_from_contour(
    contour: np.ndarray,
    offset_xy: tuple[int, int],
    boundary: Optional[LineModel],
) -> tuple[float, float]:
    pts = contour.reshape(-1, 2).astype(float)
    pts[:, 0] += offset_xy[0]
    pts[:, 1] += offset_xy[1]

    max_y = float(pts[:, 1].max())
    bottom_band = pts[pts[:, 1] >= max_y - 6.0]
    if len(bottom_band) == 0:
        bottom_band = pts

    if boundary is None:
        best = bottom_band[np.argmin(bottom_band[:, 0])]
        return float(best[0]), float(best[1])

    dists = np.array(
        [abs(float(boundary.signed_distance((float(p[0]), float(p[1]))))) for p in bottom_band],
        dtype=np.float64,
    )
    best = bottom_band[int(np.argmin(dists))]
    return float(best[0]), float(best[1])


def _candidate_blobs(
    combined_mask: np.ndarray,
    bg_mask: np.ndarray,
    thresh_mask: np.ndarray,
    roi_offset: tuple[int, int],
    min_area: float,
    boundary: Optional[LineModel],
    roi_bbox: tuple[int, int, int, int],
    cfg: dict,
) -> list[dict]:
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    x0, y0, x1, y1 = roi_bbox
    roi_h = max(1.0, float(y1 - y0))
    boundary_sigma = float(cfg.get("boundary_distance_sigma_px", 45.0))
    expected_area = float(cfg.get("expected_foot_area", 2000.0))
    min_solidity = float(cfg.get("min_solidity", 0.05))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) if len(hull) >= 3 else 0.0
        solidity = float(area / hull_area) if hull_area > 1e-6 else 0.0
        if solidity < min_solidity:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue

        contact_x, contact_y = _contact_point_from_contour(cnt, roi_offset, boundary)
        abs_boundary_dist = 999.0
        signed_boundary_dist = None
        if boundary is not None:
            signed_boundary_dist = float(boundary.signed_distance((contact_x, contact_y)))
            abs_boundary_dist = abs(signed_boundary_dist)

        patch_bg = bg_mask[by:by + bh, bx:bx + bw]
        patch_thresh = thresh_mask[by:by + bh, bx:bx + bw]
        patch_combined = combined_mask[by:by + bh, bx:bx + bw]
        denom = max(1.0, float(np.count_nonzero(patch_combined)))
        bg_overlap = float(np.count_nonzero(patch_bg)) / denom
        thresh_overlap = float(np.count_nonzero(patch_thresh)) / denom

        bottomness = float((contact_y - y0) / roi_h)
        area_conf = min(1.0, area / expected_area)
        dist_score = float(np.exp(-abs_boundary_dist / max(1.0, boundary_sigma)))
        cue_score = 0.5 * bg_overlap + 0.5 * thresh_overlap
        score = (
            0.48 * dist_score +
            0.22 * bottomness +
            0.18 * cue_score +
            0.12 * area_conf
        )

        candidates.append({
            "foot_x": round(contact_x, 2),
            "foot_y": round(contact_y, 2),
            "bbox": (bx + roi_offset[0], by + roi_offset[1], bw, bh),
            "area": area,
            "solidity": solidity,
            "bg_overlap": round(bg_overlap, 3),
            "thresh_overlap": round(thresh_overlap, 3),
            "signed_boundary_dist": round(signed_boundary_dist, 2) if signed_boundary_dist is not None else None,
            "abs_boundary_dist": round(abs_boundary_dist, 2),
            "score": round(float(score), 4),
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _select_contact_pixel(
    candidate: dict,
    combined_mask: np.ndarray,
    edge_mask: np.ndarray,
    roi_offset: tuple[int, int],
    boundary: Optional[LineModel],
) -> tuple[float, float]:
    bx, by, bw, bh = candidate["bbox"]
    lx = max(0, bx - roi_offset[0])
    ly = max(0, by - roi_offset[1])
    rx = max(lx + 1, min(combined_mask.shape[1], lx + bw))
    ry = max(ly + 1, min(combined_mask.shape[0], ly + bh))

    region = combined_mask[ly:ry, lx:rx]
    edges = edge_mask[ly:ry, lx:rx]
    if region.size == 0:
        return float(candidate["foot_x"]), float(candidate["foot_y"])

    edge_pts = np.column_stack(np.where(edges > 0))
    if len(edge_pts) == 0:
        edge_pts = np.column_stack(np.where(region > 0))
    if len(edge_pts) == 0:
        return float(candidate["foot_x"]), float(candidate["foot_y"])

    # Convert to full-frame (x, y) points.
    pts = np.stack(
        [
            edge_pts[:, 1].astype(np.float64) + bx,
            edge_pts[:, 0].astype(np.float64) + by,
        ],
        axis=1,
    )
    max_y = float(pts[:, 1].max())
    bottom_band = pts[pts[:, 1] >= max_y - 6.0]
    if len(bottom_band) == 0:
        bottom_band = pts

    if boundary is None:
        best = bottom_band[int(np.argmax(bottom_band[:, 1]))]
        return float(best[0]), float(best[1])

    dists = np.array([abs(float(boundary.signed_distance((p[0], p[1])))) for p in bottom_band], dtype=np.float64)
    idx = int(np.argmin(dists))
    best = bottom_band[idx]
    return float(best[0]), float(best[1])


def _refine_contact_point_near_seed(
    combined_mask: np.ndarray,
    edge_mask: np.ndarray,
    roi_offset: tuple[int, int],
    seed_point: tuple[float, float],
    boundary: Optional[LineModel],
    cfg: dict,
) -> tuple[float, float, tuple[int, int, int, int]]:
    ox, oy = roi_offset
    sx = float(seed_point[0]) - ox
    sy = float(seed_point[1]) - oy
    if combined_mask.size == 0:
        return float(seed_point[0]), float(seed_point[1]), (int(seed_point[0]) - 12, int(seed_point[1]) - 12, 24, 24)

    foot_half_width = int(cfg.get("pose_contact_half_width", 42))
    foot_above = int(cfg.get("pose_contact_above_px", 26))
    foot_below = int(cfg.get("pose_contact_below_px", 58))

    x0 = max(0, int(round(sx)) - foot_half_width)
    x1 = min(combined_mask.shape[1], int(round(sx)) + foot_half_width)
    y0 = max(0, int(round(sy)) - foot_above)
    y1 = min(combined_mask.shape[0], int(round(sy)) + foot_below)
    local_mask = combined_mask[y0:y1, x0:x1]
    local_edges = edge_mask[y0:y1, x0:x1]

    pts = np.column_stack(np.where(local_edges > 0))
    if len(pts) == 0:
        pts = np.column_stack(np.where(local_mask > 0))
    if len(pts) == 0:
        return float(seed_point[0]), float(seed_point[1]), (ox + x0, oy + y0, max(1, x1 - x0), max(1, y1 - y0))

    full_pts = np.stack(
        [
            pts[:, 1].astype(np.float64) + x0 + ox,
            pts[:, 0].astype(np.float64) + y0 + oy,
        ],
        axis=1,
    )
    max_y = float(full_pts[:, 1].max())
    bottom_band = full_pts[full_pts[:, 1] >= max_y - 5.0]
    if len(bottom_band) == 0:
        bottom_band = full_pts

    if boundary is None:
        best = bottom_band[int(np.argmax(bottom_band[:, 1]))]
    else:
        dists = np.array([abs(float(boundary.signed_distance((p[0], p[1])))) for p in bottom_band], dtype=np.float64)
        best = bottom_band[int(np.argmin(dists))]
    return float(best[0]), float(best[1]), (ox + x0, oy + y0, max(1, x1 - x0), max(1, y1 - y0))


def _smooth_event_candidates(
    detections: list[dict],
    target_pos: int,
    cfg: dict,
) -> Optional[dict]:
    valid = [d for d in detections if d is not None]
    if not valid:
        return None

    sigma = float(cfg.get("temporal_sigma_frames", 1.0))
    min_support = int(cfg.get("min_temporal_support", 2))
    low_conf_thresh = float(cfg.get("low_confidence_threshold", 0.45))

    weights = []
    xs = []
    ys = []
    scores = []
    best_det = max(valid, key=lambda d: d["score"])

    for det in valid:
        dt = abs(int(det["frame_pos"]) - target_pos)
        temporal_w = float(np.exp(-0.5 * (dt / max(0.5, sigma)) ** 2))
        weight = max(1e-3, det["score"]) * temporal_w
        weights.append(weight)
        xs.append(det["foot_x"])
        ys.append(det["foot_y"])
        scores.append(det["score"])

    stable_x = float(np.average(xs, weights=weights))
    stable_y = float(np.average(ys, weights=weights))
    support_ratio = len(valid) / max(1, len(detections))
    base_conf = float(np.average(scores, weights=weights))
    confidence = min(1.0, base_conf * (0.55 + 0.45 * support_ratio))
    low_conf = confidence < low_conf_thresh or len(valid) < min_support

    return {
        "foot_x": round(stable_x, 2),
        "foot_y": round(stable_y, 2),
        "confidence": round(confidence, 3),
        "mode": "event_hybrid",
        "bbox": best_det["bbox"],
        "roi_bbox": best_det.get("roi_bbox"),
        "low_confidence": low_conf,
        "temporal_support_n": len(valid),
        "debug": {
            "per_frame": detections,
            "target_candidate": best_det,
        },
    }


def localize_foot_event(
    frames: list[np.ndarray],
    frame_indices: list[int],
    target_index: int,
    boundary: Optional[LineModel] = None,
    cfg: Optional[dict] = None,
) -> Optional[dict]:
    """
    Localize the relevant foot on an event frame using short temporal context.

    Parameters
    ----------
    frames        : ordered list of BGR frames around the event
    frame_indices : frame numbers aligned with `frames`
    target_index  : frame index to return the stabilized detection for
    boundary      : selected NVZ boundary line for the event
    cfg           : foot-localizer config
    """
    if not frames or not frame_indices:
        return None
    if cfg is None:
        cfg = {}

    mode = str(cfg.get("mode", "event_hybrid"))
    if mode == "manual_point":
        return _manual_result(target_index, cfg)
    if mode != "event_hybrid":
        try:
            target_pos = frame_indices.index(target_index)
        except ValueError:
            target_pos = len(frames) // 2
        return localize_foot(frames[target_pos], frame_index=target_index, cfg=cfg)

    manual = _manual_result(target_index, cfg)
    if manual is not None:
        return manual

    try:
        target_pos = frame_indices.index(target_index)
    except ValueError:
        target_pos = len(frames) // 2

    roi_bbox = _boundary_roi(boundary, frames[target_pos].shape, cfg)
    x0, y0, x1, y1 = roi_bbox

    bg_history = max(len(frames) * 2, int(cfg.get("bg_history", 32)))
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=bg_history,
        varThreshold=float(cfg.get("bg_var_threshold", 30.0)),
        detectShadows=False,
    )

    detections: list[Optional[dict]] = []
    for frame_pos, frame in enumerate(frames):
        try:
            person_dets = _detect_people_pose(frame, roi_bbox, cfg)
        except Exception as e:
            logger.warning(f"Pose detection failed on frame {frame_indices[frame_pos]}: {e}")
            person_dets = []
        person_det = _select_pose_detection(person_dets, boundary, frame.shape, cfg)
        if person_det is None:
            detections.append(None)
            continue

        person_bbox = person_det["bbox"]
        leg_side, leg_points = _select_boundary_side_leg(person_det, boundary, cfg)
        lower_body_bbox = _leg_roi_from_pose(person_det, leg_side, leg_points, frame.shape, cfg)
        work_roi = lower_body_bbox
        detector_source = "pose_onnx"

        wx0, wy0, wx1, wy1 = work_roi
        roi = frame[wy0:wy1, wx0:wx1]
        if roi.size == 0:
            detections.append(None)
            continue

        fg_mask = subtractor.apply(roi)
        bg_mask = _cleanup_mask(
            fg_mask,
            int(cfg.get("morph_open_k", 3)),
            int(cfg.get("morph_close_k", 7)),
        )
        thresh_mask = _cleanup_mask(
            _threshold_mask(roi, cfg),
            int(cfg.get("morph_open_k", 3)),
            int(cfg.get("morph_close_k", 7)),
        )
        combined = cv2.bitwise_or(bg_mask, thresh_mask)
        combined = _cleanup_mask(
            combined,
            int(cfg.get("combined_open_k", 3)),
            int(cfg.get("combined_close_k", 9)),
        )
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edge_mask = cv2.Canny(gray_roi, int(cfg.get("contact_canny_low", 40)), int(cfg.get("contact_canny_high", 110)))
        edge_mask = cv2.bitwise_and(edge_mask, combined)
        conf_thresh = float(cfg.get("pose_keypoint_threshold", 0.35))
        seed_point = _pose_seed_point(person_det, leg_side, conf_thresh)
        refined_x, refined_y, foot_bbox = _refine_contact_point_near_seed(
            combined_mask=combined,
            edge_mask=edge_mask,
            roi_offset=(wx0, wy0),
            seed_point=seed_point,
            boundary=boundary,
            cfg=cfg,
        )
        foot_pt = (float(refined_x), float(refined_y))
        boundary_dist = abs(float(boundary.signed_distance(foot_pt))) if boundary is not None else 0.0
        dist_score = float(np.exp(-boundary_dist / max(1.0, float(cfg.get("boundary_distance_sigma_px", 45.0)))))
        pose_vis = _pose_leg_visibility(person_det["keypoints"], leg_side, conf_thresh)
        pose_vis_score = min(1.0, len(pose_vis) / 3.0)
        mask_density = float(np.count_nonzero(combined)) / max(1.0, float(combined.shape[0] * combined.shape[1]))
        score = 0.45 * dist_score + 0.35 * min(1.0, float(person_det["score"])) + 0.20 * pose_vis_score

        detections.append({
            "foot_x": round(refined_x, 2),
            "foot_y": round(refined_y, 2),
            "bbox": foot_bbox,
            "area": float(np.count_nonzero(combined)),
            "score": round(float(score), 4),
            "frame_index": int(frame_indices[frame_pos]),
            "frame_pos": frame_pos,
            "roi_bbox": roi_bbox,
            "work_roi_bbox": work_roi,
            "person_bbox": person_bbox,
            "lower_body_bbox": lower_body_bbox,
            "person_detections": person_dets,
            "person_detector_source": detector_source,
            "pose_keypoints": person_det["keypoints"],
            "pose_leg_side": leg_side,
            "pose_seed_point": seed_point,
            "bg_mask": bg_mask,
            "threshold_mask": thresh_mask,
            "combined_mask": combined,
            "edge_mask": edge_mask,
            "mask_density": mask_density,
        })

    smoothed = _smooth_event_candidates(detections, target_pos, cfg)
    if smoothed is None:
        return None

    target_det = detections[target_pos]
    if target_det is not None:
        smoothed["debug"]["target_candidate"] = target_det
        smoothed["roi_bbox"] = target_det["roi_bbox"]
        smoothed["bbox"] = target_det["bbox"]
        smoothed["work_roi_bbox"] = target_det.get("work_roi_bbox")
        smoothed["person_bbox"] = target_det.get("person_bbox")
        smoothed["lower_body_bbox"] = target_det.get("lower_body_bbox")
        smoothed["person_detector_source"] = target_det.get("person_detector_source")
        smoothed["pose_keypoints"] = target_det.get("pose_keypoints")
        smoothed["pose_leg_side"] = target_det.get("pose_leg_side")
        smoothed["pose_seed_point"] = target_det.get("pose_seed_point")
    return smoothed


# ── mode implementations ──────────────────────────────────────────────────────

def _localize_bg_subtraction(frame: np.ndarray, cfg: dict) -> Optional[dict]:
    subtractor = _ensure_bg_subtractor(cfg)
    fg_mask = subtractor.apply(frame)

    # morphological cleanup
    k_open = int(cfg.get("morph_open_k", 5))
    k_close = int(cfg.get("morph_close_k", 9))
    if k_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k)
    if k_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k)

    # apply ROI
    x0, y0, x1, y1 = _roi_from_cfg(cfg, frame.shape)
    roi_mask = np.zeros_like(fg_mask)
    roi_mask[y0:y1, x0:x1] = fg_mask[y0:y1, x0:x1]

    blob = _bottom_blob(roi_mask, float(cfg.get("min_blob_area", 200.0)))
    if blob is None:
        return None

    # confidence: rough function of blob area relative to expected foot region
    area_conf = min(1.0, float(blob["area"]) / float(cfg.get("expected_foot_area", 2000.0)))

    return {
        "foot_x": round(blob["cx"], 2),
        "foot_y": round(blob["foot_y"], 2),
        "confidence": round(0.4 + 0.5 * area_conf, 3),
        "mode": "background_subtraction",
        "bbox": blob["bbox"],
        "roi_bbox": (x0, y0, x1, y1),
        "low_confidence": False,
        "temporal_support_n": 1,
    }


def _localize_roi_threshold(frame: np.ndarray, cfg: dict) -> Optional[dict]:
    x0, y0, x1, y1 = _roi_from_cfg(cfg, frame.shape)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    threshold_mode = str(cfg.get("threshold_mode", "value"))
    if threshold_mode == "hsv":
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lo = np.array(cfg.get("hsv_lower", [0, 0, 0]), dtype=np.uint8)
        hi = np.array(cfg.get("hsv_upper", [180, 255, 80]), dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
    else:
        # threshold on V channel (dark regions = shadow/shoe)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = int(cfg.get("gray_threshold", 80))
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    k_close = int(cfg.get("morph_close_k", 7))
    if k_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    blob = _bottom_blob(mask, float(cfg.get("min_blob_area", 150.0)))
    if blob is None:
        return None

    # map back to full-frame coordinates
    fx = blob["cx"] + x0
    fy = blob["foot_y"] + y0
    bx, by, bw, bh = blob["bbox"]
    bbox_full = (bx + x0, by + y0, bw, bh)

    area_conf = min(1.0, float(blob["area"]) / float(cfg.get("expected_foot_area", 1500.0)))

    return {
        "foot_x": round(fx, 2),
        "foot_y": round(fy, 2),
        "confidence": round(0.35 + 0.45 * area_conf, 3),
        "mode": "roi_threshold",
        "bbox": bbox_full,
        "roi_bbox": (x0, y0, x1, y1),
        "low_confidence": False,
        "temporal_support_n": 1,
    }


# ── manual override loader ────────────────────────────────────────────────────

_manual_overrides: Optional[list[dict]] = None
_manual_overrides_path: Optional[Path] = None


def load_manual_overrides(path: Path) -> None:
    global _manual_overrides, _manual_overrides_path
    with open(path) as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else data.get("overrides", [])
    _manual_overrides = sorted(entries, key=lambda e: int(e["frame_index"]))
    _manual_overrides_path = path
    logger.info(f"Loaded {len(_manual_overrides)} manual foot overrides from {path}")


def _localize_manual(frame_index: int, cfg: dict) -> Optional[dict]:
    return _manual_result(frame_index, cfg)


# ── public API ────────────────────────────────────────────────────────────────

def localize_foot(
    frame: np.ndarray,
    frame_index: int = 0,
    cfg: Optional[dict] = None,
) -> Optional[dict]:
    """
    Locate the player's foot in a single frame.

    Parameters
    ----------
    frame       : BGR numpy array
    frame_index : frame number (used for manual override lookup)
    cfg         : config dict with at minimum {'mode': <str>}

    Returns
    -------
    dict | None  — see module docstring for field definitions.
    """
    if cfg is None:
        cfg = {}

    manual = _manual_result(frame_index, cfg)
    if manual is not None and str(cfg.get("mode", "background_subtraction")) != "manual_point":
        return manual

    mode = str(cfg.get("mode", "background_subtraction"))

    if mode == "background_subtraction":
        return _localize_bg_subtraction(frame, cfg)
    elif mode == "roi_threshold":
        return _localize_roi_threshold(frame, cfg)
    elif mode == "manual_point":
        return _localize_manual(frame_index, cfg)
    elif mode == "event_hybrid":
        return _localize_roi_threshold(frame, cfg)
    else:
        raise ValueError(f"Unknown foot_localizer mode: {mode!r}")
