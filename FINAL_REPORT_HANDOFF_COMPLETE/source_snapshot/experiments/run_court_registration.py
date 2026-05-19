"""
Phase 1: Court registration baseline.

Loads annotation JSON, fits NVZ kitchen line geometry, runs stability check,
saves per-frame line params CSV, debug overlay images, overlay video,
and a summary JSON report.

Usage:
    python experiments/run_court_registration.py \\
        --config experiments/configs/court_reg_v1.yaml

    # or override annotation path:
    python experiments/run_court_registration.py \\
        --config experiments/configs/court_reg_v1.yaml \\
        --annotations data/real/annotations/annotations.json
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.court_registration import CourtRegistration
from src.viz import draw_frame_info, draw_kitchen_lines, export_debug_frame, write_overlay_video

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def _save_reference_frame(video_path: Path, out_dir: Path, frame_idx: int = 60) -> Path:
    """Extract and save a single reference frame PNG for manual annotation."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.warning(f"Could not read frame {frame_idx} for reference export")
        return out_dir / "reference_frame.jpg"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"reference_frame_{frame_idx:05d}.jpg"
    cv2.imwrite(str(path), frame)
    logger.info(f"Reference frame saved: {path}")
    return path


def main(cfg: dict) -> None:
    video_path = Path(cfg["video"]["path"])
    annotation_path = Path(cfg["annotations"]["path"])
    results_dir = Path(cfg["output"]["results_dir"]) / cfg["run_name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = results_dir / "debug_frames"

    # ------------------------------------------------------------------
    # Guard: if no annotation file yet, save a reference frame and exit
    # ------------------------------------------------------------------
    if not annotation_path.exists():
        logger.warning(f"Annotation file not found: {annotation_path}")
        ref_path = _save_reference_frame(video_path, annotation_path.parent)
        logger.info("Run the annotation tool then re-run this script:")
        logger.info(f"  python scripts/annotate_reference.py --video {video_path} --out {annotation_path}")
        logger.info(f"  Or manually edit a copy of data/real/annotations/annotations_template.json")
        logger.info(f"  Reference frame saved at: {ref_path}")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / src_fps
    logger.info(f"Video: {video_path.name}  {W}x{H}  {src_fps:.2f}fps  "
                f"{total_frames} frames  ({duration_s:.1f}s)")

    # ------------------------------------------------------------------
    # Fit court registration
    # ------------------------------------------------------------------
    reg = CourtRegistration(annotation_path)
    reg.fit()

    # ------------------------------------------------------------------
    # Optional refinement
    # ------------------------------------------------------------------
    refinement_results = {}
    if cfg["registration"].get("refine", False):
        logger.info("Running line refinement...")
        refinement_results = reg.refine(
            cap,
            n_frames=cfg["registration"].get("refine_n_frames", 5),
            search_px=cfg["registration"].get("refine_search_px", 10),
        )

    # ------------------------------------------------------------------
    # Stability check
    # ------------------------------------------------------------------
    logger.info("Running stability check...")
    stability = reg.stability_check(cap, n_samples=cfg["registration"].get("stability_n_samples", 20))

    # ------------------------------------------------------------------
    # Per-frame line params CSV
    # ------------------------------------------------------------------
    if cfg["output"].get("save_line_params_csv", True):
        csv_path = results_dir / "line_params.csv"
        sample_row = reg.csv_row(0, 0.0)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sample_row.keys())
            writer.writeheader()
            for idx in range(total_frames):
                ts = idx / src_fps
                writer.writerow(reg.csv_row(idx, ts))
        logger.info(f"Line params CSV saved: {csv_path}  ({total_frames} rows)")

    # ------------------------------------------------------------------
    # Debug overlay frames (full resolution)
    # ------------------------------------------------------------------
    debug_indices = cfg["output"].get("debug_frame_indices", [0, 60, 300])
    for fidx in debug_indices:
        if fidx >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        ts = fidx / src_fps
        out_path = debug_dir / f"frame_{fidx:05d}.png"
        export_debug_frame(
            frame, out_path,
            registration=reg,
            frame_index=fidx,
            timestamp_s=ts,
            mark_legal=cfg["output"].get("mark_legal_zone", True),
        )

    # ------------------------------------------------------------------
    # Overlay video
    # ------------------------------------------------------------------
    if cfg["output"].get("save_overlay_video", True):
        cap.release()  # write_overlay_video opens its own capture
        overlay_path = results_dir / "overlay.mp4"
        write_overlay_video(
            video_path=video_path,
            out_path=overlay_path,
            registration=reg,
            fps=cfg["output"].get("overlay_video_fps", 10.0),
            scale=cfg["output"].get("overlay_video_scale", 0.5),
            frame_step=cfg["output"].get("overlay_frame_step", 6),
            mark_legal=cfg["output"].get("mark_legal_zone", True),
        )
    else:
        cap.release()

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    overall = "stable"
    for line_name, stats in stability.items():
        if stats.get("assessment") != "stable":
            overall = "check"

    report = {
        "run_name": cfg["run_name"],
        "video": video_path.name,
        "resolution": f"{W}x{H}",
        "fps": round(src_fps, 3),
        "total_frames": total_frames,
        "duration_s": round(duration_s, 2),
        "annotation_source": str(annotation_path),
        "lines": {
            name: line.to_dict()
            for name, line in [("near", reg.near_line), ("far", reg.far_line)]
            if line is not None
        },
        "refinement": refinement_results,
        "stability": stability,
        "overall_assessment": overall,
        "legal_side_sign": reg.legal_side_sign(),
        "outputs": {
            "line_params_csv": str(results_dir / "line_params.csv"),
            "debug_frames": str(debug_dir),
            "overlay_video": str(results_dir / "overlay.mp4"),
        },
    }

    report_path = results_dir / "summary_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Summary report saved: {report_path}")
    logger.info(f"Overall stability assessment: {overall.upper()}")
    logger.info(f"Done. All outputs in {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 court registration")
    parser.add_argument("--config", default="experiments/configs/court_reg_v1.yaml")
    parser.add_argument("--annotations", default=None,
                        help="Override annotation path from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.annotations:
        cfg["annotations"]["path"] = args.annotations
    main(cfg)
