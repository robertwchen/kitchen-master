"""
KitchenMaster Presentation Demo Pipeline
=========================================

Two-mode design for human-in-the-loop validation:

  MODE 1 — auto_review
  --------------------
  Runs all stages, exports review artifacts (annotated frames, montages, overlays),
  then writes  results/<run>/review/review_pending.json  and STOPS.

  User workflow:
    1. Inspect the artifacts in each stage's output directory.
    2. Copy review_pending.json → review_approved.json.
    3. Edit review_approved.json:
         - Set "status" on each checkpoint to "approved" or "rejected".
         - Add per-item "override" entries where the system was wrong.
    4. Re-run with mode: apply_overrides.

  MODE 2 — apply_overrides
  ------------------------
  Loads review_approved.json, applies all user corrections, then produces
  the final annotated outputs (event frames, summary video, pipeline_summary.json).

Review file schema (auto-generated; user edits status + overrides)
-------------------------------------------------------------------
{
  "run_name": "...",
  "checkpoints": {
    "registration": {
      "status": "pending",         // → "approved" | "rejected"
      "artifacts": [...],           // paths to debug frames
      "notes": "",                  // free text for user
      "override": {                 // optional manual line coords
        "kitchen_near_left":  [x, y],
        "kitchen_near_right": [x, y],
        "kitchen_far_left":   [x, y],
        "kitchen_far_right":  [x, y]
      }
    },
    "ball_tracking": {
      "status": "pending",
      "artifacts": [...],
      "notes": "",
      "override_frames": [         // list of per-frame overrides
        { "frame_index": 42, "ball_x": 512.0, "ball_y": 300.0, "confidence": 1.0 }
      ]
    },
    "bounce_candidates": {
      "status": "pending",
      "candidates": [              // one entry per detected candidate
        {
          "frame_index": 120,
          "timestamp_s": 2.0,
          "vy_before": 3.2,
          "vy_after": -2.8,
          "y_position": 640.0,
          "confidence": 0.72,
          "system_label": "bounce",
          "user_label": null,      // → "bounce" | "no_bounce" | "uncertain"
          "montage_path": "..."
        }
      ]
    },
    "foot_localizer": {
      "status": "pending",
      "events": [                  // one entry per candidate volley event
        {
          "frame_index": 120,
          "system_foot_x": 300.0,
          "system_foot_y": 950.0,
          "system_confidence": 0.7,
          "frame_path": "...",
          "override_foot_x": null, // user can set
          "override_foot_y": null
        }
      ]
    },
    "final_events": {
      "status": "pending",
      "events": [                  // one entry per evaluated event
        {
          "frame_index": 120,
          "system_label": "uncertain",
          "signed_dist_px": 3.5,
          "frame_path": "...",
          "user_label": null        // → "legal_volley" | "foot_fault_volley" | "uncertain"
        }
      ]
    }
  }
}

Usage
-----
  # Mode 1: run stages + export review artifacts
  python experiments/run_demo_pipeline.py \\
      --config experiments/configs/demo_pipeline.yaml

  # Mode 2: apply user overrides + produce final outputs
  python experiments/run_demo_pipeline.py \\
      --config experiments/configs/demo_pipeline.yaml \\
      --mode apply_overrides

  # Run only specific stages (comma-separated: 1,2,3,4,5)
  python experiments/run_demo_pipeline.py \\
      --config experiments/configs/demo_pipeline.yaml \\
      --stages 2,3

Outputs
-------
  results/presentation_demo/<run_name>/
    review/
      review_pending.json         generated after auto_review
      review_approved.json        user-edited (copy of pending, then edit)
      checkpoint_registration/    court model debug frames
      checkpoint_ball_tracking/   sampled ball tracking PNGs
      checkpoint_bounces/         3-panel montages per bounce candidate
      checkpoint_foot/            foot localizer debug frames
      checkpoint_final/           final event annotation frames
    ball_tracking/
      ball_tracking.csv
      ball_overlay.mp4
      debug_frames/
    volley_events/
      candidates.csv
      events.csv
      montage/
    foot_faults/
      foot_fault_events.csv
      event_frames/
      summary.json
    summary/
      pipeline_summary.json
      court_model_frame0.png
      demo_summary.mp4            (written only after apply_overrides)
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ball_tracker import track_ball
from src.court_model import CourtGeometryModel
from src.foot_fault_pipeline import (
    _build_zoom_panel,
    _model_from_reg_row,
    _select_boundary,
    analyze_event_feet,
    load_registration_csv,
    run_foot_fault_pipeline,
)
from src.foot_localizer import localize_foot, localize_foot_event
from src.volley_classifier import run_volley_classification
from src.viz import draw_court_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _banner(msg: str) -> None:
    sep = "─" * 64
    print(f"\n{sep}\n  {msg}\n{sep}")


def _tracking_lookup(tracking_rows: list[dict]) -> dict[int, dict]:
    return {int(r["frame_index"]): r for r in (tracking_rows or [])}


def _ball_window(
    tracking_rows: list[dict],
    center_frame: int,
    radius: int,
) -> list[dict]:
    window = []
    for row in tracking_rows or []:
        fi = int(row["frame_index"])
        if abs(fi - center_frame) > radius:
            continue
        if row.get("ball_x") is None:
            continue
        window.append({
            "frame_index": fi,
            "ball_x": float(row["ball_x"]),
            "ball_y": float(row["ball_y"]) if row.get("ball_y") is not None else None,
            "confidence": float(row.get("confidence") or 0.0),
        })
    return window


def _build_volley_events(
    volley_candidate_frames: list[int],
    tracking_rows: list[dict],
    classified_events: list[dict] | None = None,
    foot_review_events: list[dict] | None = None,
    final_review_events: list[dict] | None = None,
    ball_context_radius: int = 12,
) -> list[dict]:
    """Build event dicts with ball position plus optional review overrides."""
    tracking_by_frame = _tracking_lookup(tracking_rows)
    classified_by_frame = {int(e["frame_index"]): e for e in (classified_events or [])}
    foot_review_by_frame = {int(e["frame_index"]): e for e in (foot_review_events or [])}
    final_review_by_frame = {int(e["frame_index"]): e for e in (final_review_events or [])}

    events = []
    for fi in volley_candidate_frames:
        event = {"frame_index": int(fi), "label": "volley"}
        foot_review = foot_review_by_frame.get(int(fi), {})
        final_review = final_review_by_frame.get(int(fi), {})
        src = (
            classified_by_frame.get(int(fi)) or
            tracking_by_frame.get(int(fi)) or
            final_review or
            foot_review or
            {}
        )
        for key in ("timestamp_s", "ball_x", "ball_y", "confidence"):
            if src.get(key) is not None:
                event[key] = src.get(key)
        event["ball_window"] = _ball_window(tracking_rows, int(fi), ball_context_radius)
        event["active_side_temporal_sigma_frames"] = 6.0
        event["active_side_min_ball_confidence"] = 0.25

        override_side = (
            final_review.get("override_active_side") or
            foot_review.get("override_active_side")
        )
        if override_side in {"left", "right"}:
            event["override_active_side"] = override_side

        ox = final_review.get("override_foot_x")
        oy = final_review.get("override_foot_y")
        if ox is None or oy is None:
            ox = foot_review.get("override_foot_x")
            oy = foot_review.get("override_foot_y")
        if ox is not None and oy is not None:
            event["override_foot_x"] = float(ox)
            event["override_foot_y"] = float(oy)

        events.append(event)
    return events


def _load_tracking_rows_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame_index": int(row["frame_index"]),
                "timestamp_s": float(row["timestamp_s"]),
                "ball_x": float(row["ball_x"]) if row.get("ball_x") not in (None, "", "None") else None,
                "ball_y": float(row["ball_y"]) if row.get("ball_y") not in (None, "", "None") else None,
                "raw_ball_x": float(row["raw_ball_x"]) if row.get("raw_ball_x") not in (None, "", "None") else None,
                "raw_ball_y": float(row["raw_ball_y"]) if row.get("raw_ball_y") not in (None, "", "None") else None,
                "radius_px": row.get("radius_px"),
                "raw_confidence": float(row["raw_confidence"]) if row.get("raw_confidence") else 0.0,
                "bbox_x0": float(row["bbox_x0"]) if row.get("bbox_x0") not in (None, "", "None") else None,
                "bbox_y0": float(row["bbox_y0"]) if row.get("bbox_y0") not in (None, "", "None") else None,
                "bbox_x1": float(row["bbox_x1"]) if row.get("bbox_x1") not in (None, "", "None") else None,
                "bbox_y1": float(row["bbox_y1"]) if row.get("bbox_y1") not in (None, "", "None") else None,
                "tracking_backend": row.get("tracking_backend"),
                "confidence": float(row["confidence"]) if row.get("confidence") else 0.0,
            })
    return rows


def _load_ref_model(cfg: dict) -> CourtGeometryModel | None:
    ann_path = Path(cfg.get("annotations_path", ""))
    if not ann_path.exists():
        return None
    with open(ann_path) as f:
        ann = json.load(f)
    frames = ann.get("annotated_frames", [])
    if not frames:
        return None
    anchors = frames[0].get("anchors", {})
    try:
        return CourtGeometryModel(anchors)
    except Exception:
        return None


def _write_court_model_frame(
    video_path: Path,
    model: CourtGeometryModel,
    out_path: Path,
    frame_index: int = 0,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return
    annotated = draw_court_model(frame, model, draw_anchors=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    logger.info(f"Court model frame: {out_path}")


# ── Checkpoint 1: Registration ────────────────────────────────────────────────

def checkpoint_registration(
    cfg: dict,
    run_dir: Path,
    video_path: Path,
) -> dict:
    """
    Validate that the NVZ boundary lines are correctly positioned.
    Exports annotated debug frames at several time points.
    Returns the checkpoint dict for the review file.
    """
    _banner("Checkpoint 1: Court Registration / NVZ Line Validation")
    reg_cfg = cfg.get("registration", {})
    ckpt_dir = run_dir / "review" / "checkpoint_registration"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # load per-frame registration CSV
    csv_path_str = reg_cfg.get("csv_path", "")
    reg_rows: dict[int, dict] = {}
    if csv_path_str and Path(csv_path_str).exists():
        reg_rows = load_registration_csv(Path(csv_path_str))
        logger.info(f"  Registration CSV loaded: {len(reg_rows)} rows")
    else:
        logger.warning(f"  Registration CSV not found: {csv_path_str}")

    # load ref model from annotations
    ann_path_str = reg_cfg.get("annotations_path", "")
    ref_model = _load_ref_model({"annotations_path": ann_path_str}) if ann_path_str else None

    # select frames to inspect (spread across the clip)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 60.0
    cap.release()

    sample_indices = [0]
    if total > 0:
        sample_indices = sorted(set(
            int(i) for i in np.linspace(0, total - 1, min(8, total))
        ))

    artifacts = []
    cap = cv2.VideoCapture(str(video_path))
    for fi in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        model = None
        if fi in reg_rows:
            model = _model_from_reg_row(reg_rows[fi])
        elif reg_rows:
            nearest = min(reg_rows.keys(), key=lambda k: abs(k - fi))
            model = _model_from_reg_row(reg_rows[nearest])
        if model is None:
            model = ref_model

        annotated = frame.copy()
        if model is not None:
            annotated = draw_court_model(annotated, model, draw_anchors=True)

        ts = fi / src_fps
        label = f"f={fi}  t={ts:.1f}s  (inspect: are NVZ boundary lines correct?)"
        cv2.putText(annotated, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(annotated, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        out_path = ckpt_dir / f"frame_{fi:05d}.png"
        cv2.imwrite(str(out_path), annotated)
        artifacts.append(str(out_path))
    cap.release()

    print(f"  DEBUG FRAMES saved to: {ckpt_dir}")
    print(f"  → Inspect these frames. Are the NVZ boundary lines correctly placed?")
    print(f"  → If not, set override coordinates in review_approved.json.")

    return {
        "status": "pending",
        "artifacts": artifacts,
        "notes": "Inspect debug frames. NVZ left boundary = near-left → far-left corner. Right boundary = near-right → far-right corner.",
        "override": None,
    }


# ── Checkpoint 2: Ball tracking ───────────────────────────────────────────────

def checkpoint_ball_tracking(
    tracking_rows: list[dict],
    run_dir: Path,
    video_path: Path,
    n_samples: int = 20,
) -> dict:
    """Export sampled ball detection frames for user review."""
    _banner("Checkpoint 2: Ball Tracking Validation")
    ckpt_dir = run_dir / "review" / "checkpoint_ball_tracking"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not tracking_rows:
        print("  No tracking rows — ball tracking did not run.")
        return {"status": "pending", "artifacts": [], "notes": "Ball tracking not run.", "override_frames": []}

    n_det = sum(1 for r in tracking_rows if r.get("ball_x") is not None)
    pct = n_det / max(1, len(tracking_rows)) * 100
    print(f"  Ball detected in {n_det}/{len(tracking_rows)} frames ({pct:.1f}%)")

    # Sample frames that have detections for inspection
    detected = [r for r in tracking_rows if r.get("ball_x") is not None]
    sample_step = max(1, len(detected) // n_samples)
    sampled = detected[::sample_step][:n_samples]

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    artifacts = []
    for row in sampled:
        fi = int(row["frame_index"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        # draw detection
        bx = int(round(float(row["ball_x"])))
        by = int(round(float(row["ball_y"])))
        radius = int(round(float(row.get("radius_px") or 8)))
        cv2.circle(frame, (bx, by), max(8, radius), (0, 255, 255), 2)
        cv2.circle(frame, (bx, by), 2, (0, 255, 255), -1)
        ts = fi / src_fps
        info = (
            f"f={fi}  t={ts:.2f}s  x={row['ball_x']}  y={row['ball_y']}  "
            f"conf={row.get('confidence', '?')}"
        )
        cv2.putText(frame, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(frame, "Is this the ball? Set override_frames if incorrect.",
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(frame, "Is this the ball? Set override_frames if incorrect.",
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        out_path = ckpt_dir / f"frame_{fi:05d}.png"
        cv2.imwrite(str(out_path), frame)
        artifacts.append(str(out_path))
    cap.release()

    print(f"  Sampled detection frames saved to: {ckpt_dir}")
    print(f"  → Confirm the yellow ball is tracked correctly.")
    print(f"  → Add per-frame corrections in override_frames if needed.")

    return {
        "status": "pending",
        "detection_rate_pct": round(pct, 1),
        "artifacts": artifacts,
        "notes": "Confirm the ball is tracked correctly. Add override_frames entries to correct wrong detections.",
        "override_frames": [],
    }


# ── Checkpoint 3: Bounce candidates ──────────────────────────────────────────

def checkpoint_bounce_candidates(
    bounces: list[dict],
    run_dir: Path,
    montage_dir: Path,
) -> dict:
    """Package bounce candidates for review."""
    _banner("Checkpoint 3: Bounce / Volley Candidate Validation")
    ckpt_dir = run_dir / "review" / "checkpoint_bounces"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not bounces:
        print("  No bounce candidates found.")
        return {
            "status": "pending",
            "notes": "No bounce candidates detected. Check ball tracking coverage.",
            "candidates": [],
        }

    candidates = []
    for b in bounces:
        fi = int(b["frame_index"])
        montage_path = montage_dir / f"bounce_{fi:05d}.png"
        candidates.append({
            "frame_index": fi,
            "timestamp_s": float(b["timestamp_s"]),
            "vy_before": float(b["vy_before"]),
            "vy_after": float(b["vy_after"]),
            "y_position": float(b["ball_y"]),
            "drop_px": float(b["drop_px"]),
            "rise_px": float(b["rise_px"]),
            "near_court_surface": bool(b["near_court_surface"]),
            "confidence": float(b["confidence"]),
            "system_label": str(b["label"]),
            "user_label": None,     # user fills in: "bounce" | "no_bounce" | "uncertain"
            "montage_path": str(montage_path) if montage_path.exists() else None,
        })

    print(f"  {len(candidates)} bounce candidates found.")
    print(f"  3-panel montages in: {montage_dir}")
    print(f"  → For each candidate, set user_label to 'bounce', 'no_bounce', or 'uncertain'.")
    print(f"  → Inspect: vy_before>0 (falling) and vy_after<0 (rising) = real bounce.")

    return {
        "status": "pending",
        "notes": (
            "For each candidate, review the montage (before/at/after frames) and set user_label. "
            "Fields: vy_before (positive=ball falling), vy_after (negative=ball rising), "
            "y_position (image coords, lower y = higher in frame). "
            "A true bounce has vy_before > 0 AND vy_after < 0 near the court surface."
        ),
        "candidates": candidates,
    }


# ── Checkpoint 4: Foot localization ──────────────────────────────────────────

def checkpoint_foot_localizer(
    volley_events: list[dict],
    run_dir: Path,
    video_path: Path,
    cfg: dict,
    reg_rows: dict[int, dict],
    ref_model: CourtGeometryModel | None,
) -> dict:
    """Run foot localization on candidate frames and export debug images."""
    _banner("Checkpoint 4: Foot Localization Validation")
    ckpt_dir = run_dir / "review" / "checkpoint_foot"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not volley_events:
        print("  No candidate frames for foot localization.")
        return {
            "status": "pending",
            "notes": "No candidate frames. Check bounce detection or add manual hit_frames.",
            "events": [],
        }

    fl_cfg = cfg.get("foot_localizer", {"mode": "background_subtraction"})
    ff_cfg = cfg.get("foot_fault", {})
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    temporal_radius = int(fl_cfg.get("temporal_window_radius", 1))

    events = []
    for event in volley_events:
        fi = int(event["frame_index"])
        # draw court geometry
        model = None
        if fi in reg_rows:
            model = _model_from_reg_row(reg_rows[fi])
        elif reg_rows:
            nearest = min(reg_rows.keys(), key=lambda k: abs(k - fi))
            model = _model_from_reg_row(reg_rows[nearest])
        if model is None:
            model = ref_model

        frame_window = []
        frame_indices = []
        for sample_fi in range(max(0, fi - temporal_radius), fi + temporal_radius + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_fi)
            ret, sample_frame = cap.read()
            if not ret:
                continue
            frame_window.append(sample_frame)
            frame_indices.append(sample_fi)
        if not frame_window:
            continue
        try:
            target_pos = frame_indices.index(fi)
        except ValueError:
            target_pos = len(frame_window) // 2
        frame = frame_window[target_pos]
        out = frame.copy()
        ff_cfg_full = dict(ff_cfg)
        ff_cfg_full["foot_localizer"] = fl_cfg
        analysis = analyze_event_feet(
            event=event,
            frame=frame,
            frames=frame_window,
            frame_indices=frame_indices,
            frame_index=fi,
            model=model,
            cfg=ff_cfg_full,
        )
        active_side = analysis.get("active_side")
        primary = analysis.get("side_results", {}).get(active_side, {})
        foot = primary.get("foot_result")
        boundary = primary.get("boundary")

        if model is not None:
            out = draw_court_model(out, model, draw_anchors=False)

        H, W = out.shape[:2]
        for side_name, color in (("left", (255, 120, 0)), ("right", (180, 0, 255))):
            side_result = analysis.get("side_results", {}).get(side_name, {})
            boundary_side = side_result.get("boundary")
            if boundary_side is not None:
                pt1, pt2 = boundary_side.endpoints_in_frame(W, H)
                line_color = (0, 255, 255) if side_name == active_side else (0, 220, 50)
                cv2.line(out, pt1, pt2, line_color, 3 if side_name == active_side else 2)
            foot_side = side_result.get("foot_result")
            if foot_side is None:
                continue
            person_bbox = foot_side.get("person_bbox")
            if person_bbox is not None:
                px, py, pw, ph = [int(v) for v in person_bbox]
                cv2.rectangle(out, (px, py), (px + pw, py + ph), (80, 220, 255), 2 if side_name == active_side else 1)
            pose_keypoints = foot_side.get("pose_keypoints")
            if pose_keypoints is not None:
                for kp_idx in (11, 12, 13, 14, 15, 16):
                    kp = pose_keypoints[kp_idx]
                    if float(kp[2]) < 0.35:
                        continue
                    cv2.circle(out, (int(round(float(kp[0]))), int(round(float(kp[1])))), 4, color, -1)
            lower_body_bbox = foot_side.get("lower_body_bbox")
            if lower_body_bbox is not None:
                lx0, ly0, lx1, ly1 = [int(v) for v in lower_body_bbox]
                cv2.rectangle(out, (lx0, ly0), (lx1, ly1), (255, 255, 0), 2 if side_name == active_side else 1)
            fx = int(round(float(foot_side["foot_x"])))
            fy = int(round(float(foot_side["foot_y"])))
            cv2.circle(out, (fx, fy), 10 if side_name == active_side else 7, color, -1)
            cv2.circle(out, (fx, fy), 10 if side_name == active_side else 7, (255, 255, 255), 2 if side_name == active_side else 1)
            bbox = foot_side.get("bbox")
            if bbox:
                bx, by, bw, bh = bbox
                cv2.rectangle(out, (int(bx), int(by)), (int(bx+bw), int(by+bh)), color, 2 if side_name == active_side else 1)

        signed_dist = analysis.get("signed_dist_px")
        preview_label = analysis.get("label", "uncertain")
        left_dist = analysis.get("side_results", {}).get("left", {}).get("signed_dist_px")
        right_dist = analysis.get("side_results", {}).get("right", {}).get("signed_dist_px")

        # draw foot result
        ts = fi / src_fps
        if foot is not None:
            info = (
                f"f={fi}  t={ts:.2f}s  active={active_side} ({analysis.get('active_side_source')}, "
                f"conf={analysis.get('side_confidence', 0.0):.2f}, n={analysis.get('ball_support_n', 0)})  "
                f"foot=({foot['foot_x']:.0f},{foot['foot_y']:.0f})  conf={foot['confidence']:.2f}"
            )
        else:
            info = (
                f"f={fi}  t={ts:.2f}s  active={active_side} ({analysis.get('active_side_source')}, "
                f"conf={analysis.get('side_confidence', 0.0):.2f}, n={analysis.get('ball_support_n', 0)})  "
                "NO FOOT DETECTED"
            )
        left_str = f"{left_dist:+.1f}" if left_dist is not None else "N/A"
        right_str = f"{right_dist:+.1f}" if right_dist is not None else "N/A"
        info2 = f"left_dist={left_str}px  right_dist={right_str}px  pred={preview_label}"

        cv2.putText(out, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(out, info2, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
        cv2.putText(out, info2, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 255, 255), 1)
        cv2.putText(out, "cyan=person  yellow=lower-body  colored dot=chosen contact point",
                    (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, "cyan=person  yellow=lower-body  colored dot=chosen contact point",
                    (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
        cv2.putText(out, "Check active side, boundary choice, and contact point. Set overrides if wrong.",
                    (8, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, "Check active side, boundary choice, and contact point. Set overrides if wrong.",
                    (8, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

        zoom = _build_zoom_panel(frame, foot, boundary, title=f"{active_side.title()} ROI" if active_side else "Foot ROI")
        zoom_path = ckpt_dir / f"foot_{fi:05d}_roi.png"
        cv2.imwrite(str(zoom_path), zoom)

        out_path = ckpt_dir / f"foot_{fi:05d}.png"
        cv2.imwrite(str(out_path), out)

        events.append({
            "frame_index": fi,
            "active_side": active_side,
            "active_side_source": analysis.get("active_side_source"),
            "active_side_confidence": round(float(analysis.get("side_confidence", 0.0)), 3),
            "ball_support_n": int(analysis.get("ball_support_n", 0)),
            "ball_x": round(float(analysis["ball_x"]), 2) if analysis.get("ball_x") is not None else None,
            "ball_y": round(float(analysis["ball_y"]), 2) if analysis.get("ball_y") is not None else None,
            "predicted_label": preview_label,
            "signed_dist_px": round(signed_dist, 2) if signed_dist is not None else None,
            "left_signed_dist_px": round(left_dist, 2) if left_dist is not None else None,
            "right_signed_dist_px": round(right_dist, 2) if right_dist is not None else None,
            "system_foot_x": round(float(foot["foot_x"]), 2) if foot else None,
            "system_foot_y": round(float(foot["foot_y"]), 2) if foot else None,
            "system_confidence": round(float(foot["confidence"]), 3) if foot else None,
            "frame_path": str(out_path),
            "roi_frame_path": str(zoom_path),
            "override_active_side": None,
            "override_foot_x": None,   # user fills in if incorrect
            "override_foot_y": None,
        })

    cap.release()
    print(f"  Foot localization frames saved to: {ckpt_dir}")
    print(f"  → Inspect each frame. If the foot point is wrong, set override_foot_x/y.")
    print(f"  → Tip: use mode: manual_point in config to set all points manually.")

    return {
        "status": "pending",
        "notes": (
            "Review each foot localization frame. If the wrong player is being judged, set "
            "override_active_side to left or right. Set override_foot_x / override_foot_y "
            "to correct the active-side contact point. Validate the selected boundary, both-side "
            "distances, the active-side ROI, and whether the signed-distance sign makes sense."
        ),
        "events": events,
    }


# ── Checkpoint 5: Final event validation ─────────────────────────────────────

def checkpoint_final_events(
    fault_results: list[dict],
    run_dir: Path,
) -> dict:
    """Package final event frames for user approval."""
    _banner("Checkpoint 5: Final Event Validation")
    ckpt_dir = run_dir / "review" / "checkpoint_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not fault_results:
        print("  No fault events to review.")
        return {
            "status": "pending",
            "notes": "No foot-fault events evaluated.",
            "events": [],
        }

    events = []
    for r in fault_results:
        # copy annotated frame to checkpoint dir for easy inspection
        src = Path(r.get("frame_path", ""))
        dst = ckpt_dir / src.name if src.exists() else None
        if src.exists() and dst:
            import shutil
            shutil.copy2(src, dst)
            frame_path_ckpt = str(dst)
        else:
            frame_path_ckpt = r.get("frame_path")

        events.append({
            "frame_index": r["frame_index"],
            "timestamp_s": r["timestamp_s"],
            "side": r.get("side"),
            "active_side": r.get("active_side"),
            "inferred_active_side": r.get("inferred_active_side"),
            "active_side_source": r.get("active_side_source"),
            "active_side_confidence": r.get("active_side_confidence"),
            "ball_support_n": r.get("ball_support_n"),
            "ball_x": r.get("ball_x"),
            "ball_y": r.get("ball_y"),
            "signed_dist_px": r.get("signed_dist_px"),
            "left_signed_dist_px": r.get("left_signed_dist_px"),
            "right_signed_dist_px": r.get("right_signed_dist_px"),
            "system_label": r["label"],
            "foot_x": r.get("foot_x"),
            "foot_y": r.get("foot_y"),
            "foot_confidence": r.get("foot_confidence"),
            "frame_path": frame_path_ckpt,
            "roi_frame_path": r.get("roi_frame_path"),
            "override_active_side": None,
            "user_label": None,  # → "legal_volley" | "foot_fault_volley" | "uncertain" | null (accept system)
        })

    print(f"  {len(events)} events ready for final review: {ckpt_dir}")
    print(f"  → For each event, check the annotated frame.")
    print(f"  → Set user_label to override the system label, or leave null to accept it.")

    return {
        "status": "pending",
        "notes": (
            "Review each event frame together with its ROI crop. The annotated frame shows: "
            "the inferred active side, both boundary distances, active-side foot point, and system label. "
            "If the wrong player is being judged, set override_active_side to left or right. "
            "Positive signed_dist = foot behind line (legal). Negative signed_dist = foot in kitchen (fault). "
            "Set user_label to override, or leave null to accept the system decision."
        ),
        "events": events,
    }


# ── Write / load review file ──────────────────────────────────────────────────

def write_review_pending(review_data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(review_data, f, indent=2)
    logger.info(f"Review file written: {path}")


def load_review_approved(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── auto_review mode ─────────────────────────────────────────────────────────

def run_auto_review(cfg: dict, run_dir: Path, video_path: Path, stages: set[int]) -> None:
    reg_cfg = cfg.get("registration", {})
    clip_start = int(cfg["video"].get("clip_start_frame", 0))
    clip_end_raw = cfg["video"].get("clip_end_frame")
    clip_end = int(clip_end_raw) if clip_end_raw is not None else None

    # --- Stage 1: Registration checkpoint ---
    s1_ckpt = {}
    if 1 in stages:
        s1_ckpt = checkpoint_registration(cfg, run_dir, video_path)

    # --- Stage 2: Ball tracking ---
    tracking_rows: list[dict] = []
    if 2 in stages:
        bt_cfg = cfg.get("ball_tracking", {})
        if bt_cfg.get("enabled", True):
            out_dir = run_dir / "ball_tracking"
            tracking_rows = track_ball(
                video_path=video_path,
                output_dir=out_dir,
                cfg=bt_cfg,
                clip_start_frame=clip_start,
                clip_end_frame=clip_end,
                debug_every_n=int(bt_cfg.get("debug_every_n", 60)),
                write_overlay=bool(bt_cfg.get("write_overlay", True)),
                overlay_fps=float(bt_cfg.get("overlay_fps", 10.0)),
                overlay_scale=float(bt_cfg.get("overlay_scale", 0.5)),
            )
    if not tracking_rows:
        tracking_rows = _load_tracking_rows_csv(run_dir / "ball_tracking" / "ball_tracking.csv")

    s2_ckpt = checkpoint_ball_tracking(tracking_rows, run_dir, video_path)

    # --- Stage 3: Bounce/volley ---
    vc_result = {"bounces": [], "events": []}
    if 3 in stages:
        vc_cfg = cfg.get("volley_classification", {})
        if vc_cfg.get("enabled", True):
            # get court surface band from registration reference geometry
            ann_path_str = reg_cfg.get("annotations_path", "")
            court_surface_y = None
            if ann_path_str:
                model = _load_ref_model({"annotations_path": ann_path_str})
                if model is not None and model.near_kitchen_line is not None:
                    near_p1 = model.near_kitchen_line.p1
                    near_p2 = model.near_kitchen_line.p2
                    near_y = float((near_p1[1] + near_p2[1]) / 2.0)
                    if model.far_kitchen_line is not None:
                        far_p1 = model.far_kitchen_line.p1
                        far_p2 = model.far_kitchen_line.p2
                        far_y = float((far_p1[1] + far_p2[1]) / 2.0)
                        court_surface_y = (far_y, near_y)
                    else:
                        court_surface_y = near_y

            hit_frames = [int(x) for x in vc_cfg.get("hit_frames", []) or []]
            vc_result = run_volley_classification(
                tracking_rows=tracking_rows,
                video_path=video_path,
                output_dir=run_dir / "volley_events",
                cfg=vc_cfg,
                court_surface_y=court_surface_y,
                hit_frames=hit_frames if hit_frames else None,
            )

    s3_ckpt = checkpoint_bounce_candidates(
        vc_result["bounces"],
        run_dir,
        run_dir / "volley_events" / "montage",
    )

    # --- Build volley candidate list ---
    manual_volley = cfg.get("foot_fault", {}).get("manual_volley_frames") or []
    if manual_volley:
        volley_candidate_frames = [int(f) for f in manual_volley]
    elif vc_result.get("events"):
        volley_candidate_frames = [
            e["frame_index"] for e in vc_result["events"] if e["label"] == "volley"
        ]
    else:
        volley_candidate_frames = [
            b["frame_index"] for b in vc_result["bounces"] if b.get("label") == "bounce"
        ]

    # --- Stage 4: Foot localization checkpoint ---
    reg_rows: dict[int, dict] = {}
    csv_path_str = reg_cfg.get("csv_path", "")
    if csv_path_str and Path(csv_path_str).exists():
        reg_rows = load_registration_csv(Path(csv_path_str))
    ann_path_str = reg_cfg.get("annotations_path", "")
    ref_model = _load_ref_model({"annotations_path": ann_path_str}) if ann_path_str else None

    volley_events = _build_volley_events(
        volley_candidate_frames=volley_candidate_frames,
        tracking_rows=tracking_rows,
        classified_events=vc_result.get("events"),
        ball_context_radius=int(cfg.get("foot_fault", {}).get("active_side_window_frames", 12)),
    )

    s4_ckpt = checkpoint_foot_localizer(
        volley_events, run_dir, video_path, cfg, reg_rows, ref_model
    )

    # --- Stage 5: Run foot fault for review ---
    fault_results: list[dict] = []
    if (4 in stages or 5 in stages) and volley_candidate_frames:
        ff_cfg = cfg.get("foot_fault", {})
        fl_cfg = cfg.get("foot_localizer", {})
        ff_cfg_full = dict(ff_cfg)
        ff_cfg_full["foot_localizer"] = fl_cfg
        manual_override_str = reg_cfg.get("manual_override_path", "")
        manual_override = Path(manual_override_str) if manual_override_str else None
        ann_path = Path(ann_path_str) if ann_path_str else None

        fault_results = run_foot_fault_pipeline(
            volley_events=volley_events,
            video_path=video_path,
            output_dir=run_dir / "foot_faults",
            cfg=ff_cfg_full,
            registration_csv=Path(csv_path_str) if csv_path_str else None,
            manual_line_override_path=manual_override,
            ref_annotations_path=ann_path,
        )

    s5_ckpt = checkpoint_final_events(fault_results, run_dir)

    # --- Write review_pending.json ---
    review_data = {
        "run_name": cfg["run_name"],
        "pipeline_mode": "auto_review",
        "instructions": (
            "1. Inspect artifacts in each checkpoint directory.\n"
            "2. Copy this file to review_approved.json.\n"
            "3. Set 'status' to 'approved' for each checkpoint (or 'rejected' if rerun needed).\n"
            "4. Fill in user_label / override fields where the system was wrong.\n"
            "5. Rerun with --mode apply_overrides."
        ),
        "checkpoints": {
            "registration": s1_ckpt,
            "ball_tracking": s2_ckpt,
            "bounce_candidates": s3_ckpt,
            "foot_localizer": s4_ckpt,
            "final_events": s5_ckpt,
        },
    }

    pipeline_cfg = cfg.get("pipeline", {})
    review_pending_path = Path(pipeline_cfg.get(
        "review_pending_path",
        str(run_dir / "review" / "review_pending.json"),
    ))
    write_review_pending(review_data, review_pending_path)

    _banner("auto_review complete — NEXT STEPS")
    print(f"\n  Review artifacts are in: {run_dir / 'review'}/")
    print(f"\n  Review file: {review_pending_path}")
    print(f"\n  To validate and produce final outputs:")
    print(f"    1. Inspect each checkpoint_*/  directory")
    print(f"    2. Copy:  cp {review_pending_path} \\")
    print(f"              {str(review_pending_path).replace('pending', 'approved')}")
    print(f"    3. Edit review_approved.json:")
    print(f"         - Set each checkpoint's 'status' to 'approved'")
    print(f"         - Add user_label / override where the system was wrong")
    print(f"    4. Rerun:")
    print(f"       python experiments/run_demo_pipeline.py \\")
    print(f"           --config experiments/configs/demo_pipeline.yaml \\")
    print(f"           --mode apply_overrides\n")


# ── apply_overrides mode ──────────────────────────────────────────────────────

def _apply_ball_overrides(
    tracking_rows: list[dict],
    override_frames: list[dict],
) -> list[dict]:
    """Merge user-supplied per-frame ball position corrections."""
    overrides = {int(o["frame_index"]): o for o in (override_frames or [])}
    result = []
    for row in tracking_rows:
        fi = row["frame_index"]
        if fi in overrides:
            ov = overrides[fi]
            row = dict(row)
            row["ball_x"] = ov.get("ball_x", row["ball_x"])
            row["ball_y"] = ov.get("ball_y", row["ball_y"])
            row["confidence"] = float(ov.get("confidence", 1.0))
        result.append(row)
    return result


def _apply_bounce_overrides(bounces: list[dict], candidates_review: list[dict]) -> list[dict]:
    """Apply user_label corrections to bounce candidates."""
    label_map = {int(c["frame_index"]): c.get("user_label") for c in candidates_review}
    result = []
    for b in bounces:
        fi = int(b["frame_index"])
        user = label_map.get(fi)
        if user is not None:
            b = dict(b)
            b["label"] = user
        result.append(b)
    return result


def run_apply_overrides(cfg: dict, run_dir: Path, video_path: Path) -> None:
    pipeline_cfg = cfg.get("pipeline", {})
    approved_path = Path(pipeline_cfg.get(
        "review_approved_path",
        str(run_dir / "review" / "review_approved.json"),
    ))

    if not approved_path.exists():
        print(f"\n  ERROR: review_approved.json not found: {approved_path}")
        print(f"  Run auto_review first, then copy and edit review_pending.json.\n")
        sys.exit(1)

    _banner("apply_overrides: Loading User-Approved Review File")
    review = load_review_approved(approved_path)
    ckpts = review.get("checkpoints", {})

    def _ckpt_approved(key: str) -> bool:
        return ckpts.get(key, {}).get("status") == "approved"

    if not _ckpt_approved("registration"):
        print("  WARNING: Registration checkpoint not approved. Using system registration.")
    if not _ckpt_approved("ball_tracking"):
        print("  WARNING: Ball tracking checkpoint not approved. Using system detections.")
    if not _ckpt_approved("bounce_candidates"):
        print("  WARNING: Bounce candidates checkpoint not approved. Using system labels.")
    if not _ckpt_approved("foot_localizer"):
        print("  WARNING: Foot localizer checkpoint not approved. Using system detections.")
    if not _ckpt_approved("final_events"):
        print("  WARNING: Final events checkpoint not approved. Using system labels.")

    # --- Registration override ---
    reg_override = ckpts.get("registration", {}).get("override")
    manual_line_model = None
    if reg_override:
        try:
            from src.court_model import CourtGeometryModel
            manual_line_model = CourtGeometryModel(reg_override)
            logger.info("Applied user registration override")
        except Exception as e:
            logger.warning(f"Could not apply registration override: {e}")

    # --- Ball tracking rows: reload CSV and apply frame overrides ---
    bt_csv = run_dir / "ball_tracking" / "ball_tracking.csv"
    tracking_rows: list[dict] = []
    if bt_csv.exists():
        tracking_rows = _load_tracking_rows_csv(bt_csv)
    ball_ov_frames = ckpts.get("ball_tracking", {}).get("override_frames", [])
    tracking_rows = _apply_ball_overrides(tracking_rows, ball_ov_frames)

    # --- Bounce candidates: reload and apply user labels ---
    cand_csv = run_dir / "volley_events" / "candidates.csv"
    bounces: list[dict] = []
    if cand_csv.exists():
        with open(cand_csv, newline="") as f:
            for row in csv.DictReader(f):
                bounces.append({
                    "frame_index": int(row["frame_index"]),
                    "timestamp_s": float(row["timestamp_s"]),
                    "ball_y": float(row["ball_y"]),
                    "ball_x": float(row["ball_x"]),
                    "vy_before": float(row["vy_before"]),
                    "vy_after": float(row["vy_after"]),
                    "drop_px": float(row["drop_px"]),
                    "rise_px": float(row["rise_px"]),
                    "near_court_surface": row["near_court_surface"] == "True",
                    "confidence": float(row["confidence"]),
                    "label": row["label"],
                })
    bounce_cands_review = ckpts.get("bounce_candidates", {}).get("candidates", [])
    bounces = _apply_bounce_overrides(bounces, bounce_cands_review)

    # --- Build confirmed volley frames ---
    foot_ov_frames = ckpts.get("foot_localizer", {}).get("events", [])

    manual_volley = cfg.get("foot_fault", {}).get("manual_volley_frames") or []
    final_events_review = ckpts.get("final_events", {}).get("events", [])
    user_labels = {int(e["frame_index"]): e.get("user_label") for e in final_events_review}

    if manual_volley:
        volley_candidate_frames = [int(f) for f in manual_volley]
    else:
        volley_candidate_frames = [
            b["frame_index"] for b in bounces if b.get("label") == "bounce"
        ]

    # --- Re-run foot fault with overrides ---
    reg_cfg = cfg.get("registration", {})
    csv_path_str = reg_cfg.get("csv_path", "")
    ann_path_str = reg_cfg.get("annotations_path", "")
    ff_cfg = cfg.get("foot_fault", {})
    fl_cfg = cfg.get("foot_localizer", {})

    ff_cfg_full = dict(ff_cfg)
    ff_cfg_full["foot_localizer"] = fl_cfg
    volley_events = _build_volley_events(
        volley_candidate_frames=volley_candidate_frames,
        tracking_rows=tracking_rows,
        classified_events=[],
        foot_review_events=foot_ov_frames,
        final_review_events=final_events_review,
        ball_context_radius=int(cfg.get("foot_fault", {}).get("active_side_window_frames", 12)),
    )
    fault_results = run_foot_fault_pipeline(
        volley_events=volley_events,
        video_path=video_path,
        output_dir=run_dir / "foot_faults_final",
        cfg=ff_cfg_full,
        registration_csv=Path(csv_path_str) if csv_path_str else None,
        manual_line_override_path=None,
        ref_annotations_path=Path(ann_path_str) if ann_path_str else None,
    )

    # Apply user label overrides on top of system decisions
    for r in fault_results:
        fi = int(r["frame_index"])
        ul = user_labels.get(fi)
        if ul:
            r["label"] = ul
            r["user_override"] = True

    # --- Write summary video ---
    reg_rows: dict[int, dict] = {}
    if csv_path_str and Path(csv_path_str).exists():
        reg_rows = load_registration_csv(Path(csv_path_str))
    _write_summary_video(cfg, run_dir, video_path, tracking_rows, fault_results, reg_rows)

    # Pipeline summary
    label_counts = Counter(r["label"] for r in fault_results)
    summary = {
        "run_name": cfg["run_name"],
        "mode": "apply_overrides",
        "review_file": str(approved_path),
        "n_tracking_rows": len(tracking_rows),
        "n_bounce_candidates": len(bounces),
        "n_volley_events": len(volley_candidate_frames),
        "foot_fault_labels": dict(label_counts),
        "output_dir": str(run_dir),
    }
    summary_path = run_dir / "summary" / "pipeline_summary_final.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _banner("apply_overrides complete")
    print(f"\n  Final results: {run_dir}/foot_faults_final/")
    print(f"  Summary video: {run_dir}/summary/demo_summary.mp4")
    print(f"  Labels: {dict(label_counts)}")
    print(f"  Summary: {summary_path}\n")


# ── Summary video (shared) ────────────────────────────────────────────────────

def _write_summary_video(
    cfg: dict,
    run_dir: Path,
    video_path: Path,
    tracking_rows: list[dict],
    fault_results: list[dict],
    reg_rows: dict[int, dict],
) -> None:
    out_cfg = cfg.get("output", {})
    if not out_cfg.get("write_summary_video", True):
        return

    tracking_by_frame = {r["frame_index"]: r for r in tracking_rows}
    fault_by_frame = {r["frame_index"]: r for r in fault_results}

    ann_path_str = cfg.get("registration", {}).get("annotations_path", "")
    fallback_model = _load_ref_model({"annotations_path": ann_path_str}) if ann_path_str else None

    scale = float(out_cfg.get("summary_video_scale", 0.5))
    fps = float(out_cfg.get("summary_video_fps", 10.0))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_W, out_H = int(src_W * scale), int(src_H * scale)

    out_path = run_dir / "summary" / "demo_summary.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_W, out_H))

    frame_step = max(1, int(round(src_fps / fps)))
    trail: list[tuple[float, float] | None] = []
    trail_len = 12
    written = 0
    fi = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if fi % frame_step == 0:
            small = cv2.resize(frame, (out_W, out_H))

            # Court model overlay
            model = None
            if fi in reg_rows:
                model = _model_from_reg_row(reg_rows[fi])
            elif reg_rows:
                nearest = min(reg_rows.keys(), key=lambda k: abs(k - fi))
                model = _model_from_reg_row(reg_rows[nearest])
            if model is None:
                model = fallback_model
            if model is not None:
                small = draw_court_model(small, model, draw_anchors=False)

            # Ball trail
            brow = tracking_by_frame.get(fi)
            bx = float(brow["ball_x"]) * scale if (brow and brow.get("ball_x") is not None) else None
            by_v = float(brow["ball_y"]) * scale if (brow and brow.get("ball_y") is not None) else None
            trail.append((bx, by_v) if bx is not None else None)
            if len(trail) > trail_len:
                trail.pop(0)

            for k, pt in enumerate(trail):
                if pt is None:
                    continue
                alpha = (k + 1) / max(1, len(trail))
                c = (int(20 * alpha), int(200 * alpha), int(255 * alpha))
                cv2.circle(small, (int(pt[0]), int(pt[1])), 3, c, -1)
            if bx is not None:
                cv2.circle(small, (int(bx), int(by_v)), 8, (0, 255, 255), 2)

            # Fault event annotation
            if fi in fault_by_frame:
                frow = fault_by_frame[fi]
                label = frow["label"]
                color = {
                    "foot_fault_volley": (0, 60, 220),
                    "legal_volley": (0, 180, 60),
                }.get(label, (0, 180, 220))
                txt = f"★ {label.upper()} ★"
                cv2.putText(small, txt, (out_W // 4, out_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
                cv2.putText(small, txt, (out_W // 4, out_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                dist_str = f"dist={frow.get('signed_dist_px')}px"
                cv2.putText(small, dist_str, (out_W // 4, out_H // 2 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(small, dist_str, (out_W // 4, out_H // 2 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            ts = fi / src_fps
            cv2.putText(small, f"f={fi}  t={ts:.2f}s",
                        (8, out_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(small, f"f={fi}  t={ts:.2f}s",
                        (8, out_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            writer.write(small)
            written += 1

        fi += 1

    cap.release()
    writer.release()
    logger.info(f"Summary video: {out_path}  ({written} frames)")
    print(f"\n  Summary video: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KitchenMaster demo pipeline (Stages 1–5, human-in-the-loop)"
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/demo_pipeline.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--mode",
        choices=["auto_review", "apply_overrides"],
        default=None,
        help="Pipeline mode override (overrides config pipeline.mode if given)",
    )
    parser.add_argument(
        "--stages",
        default="1,2,3,4,5",
        help="Comma-separated stage numbers to run in auto_review mode (default: 1,2,3,4,5)",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    run_name = cfg["run_name"]
    video_path = Path(cfg["video"]["path"])
    out_cfg = cfg.get("output", {})
    run_dir = Path(out_cfg.get("results_dir", "results/presentation_demo/")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    pipeline_cfg = cfg.get("pipeline", {})
    mode = args.mode or pipeline_cfg.get("mode", "auto_review")
    stages = {int(s.strip()) for s in args.stages.split(",")}

    print(f"\n{'═'*64}")
    print(f"  KitchenMaster Demo Pipeline")
    print(f"  run_name : {run_name}")
    print(f"  video    : {video_path}")
    print(f"  output   : {run_dir}")
    print(f"  mode     : {mode}")
    if mode == "auto_review":
        print(f"  stages   : {sorted(stages)}")
    print(f"{'═'*64}")

    if mode == "auto_review":
        run_auto_review(cfg, run_dir, video_path, stages)
    elif mode == "apply_overrides":
        run_apply_overrides(cfg, run_dir, video_path)
    else:
        logger.error(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
