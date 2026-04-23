"""
Extract frames from a video at a configurable rate and save a manifest CSV.

Usage:
    python scripts/extract_frames.py \\
        --video data/real/videos/pickle_vid_1.MOV \\
        --out   data/real/frames/ \\
        --fps   5

Outputs:
    data/real/frames/<stem>_frame<NNNNN>.jpg   — extracted frames
    data/real/frames/manifest.csv              — frame_index, timestamp_s, filename
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def extract_frames(
    video_path: Path,
    out_dir: Path,
    target_fps: float,
    img_quality: int = 90,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / target_fps))

    logger.info(f"Video: {video_path.name}  |  {src_fps:.2f}fps  |  {total_frames} frames")
    logger.info(f"Extracting every {step}th frame → ~{target_fps:.1f}fps output")

    stem = video_path.stem
    manifest_path = out_dir / "manifest.csv"
    saved = 0

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=["frame_index", "timestamp_s", "filename"])
        writer.writeheader()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                ts = frame_idx / src_fps
                fname = f"{stem}_frame{frame_idx:05d}.jpg"
                out_path = out_dir / fname
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, img_quality])
                writer.writerow({
                    "frame_index": frame_idx,
                    "timestamp_s": round(ts, 4),
                    "filename": fname,
                })
                saved += 1

            frame_idx += 1

    cap.release()
    logger.info(f"Saved {saved} frames to {out_dir}/")
    logger.info(f"Manifest written to {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="data/real/frames/", help="Output directory")
    parser.add_argument("--fps", type=float, default=5.0, help="Target extraction FPS")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality 0–100")
    args = parser.parse_args()

    extract_frames(
        video_path=Path(args.video),
        out_dir=Path(args.out),
        target_fps=args.fps,
        img_quality=args.quality,
    )
