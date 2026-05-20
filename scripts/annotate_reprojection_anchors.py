"""
Interactive anchor annotation tool for reprojection validation.

Annotate the same court anchors on 10-20 sampled frames so court_reg_v3 can
measure projected-vs-labeled pixel error.

Usage
-----
python scripts/annotate_reprojection_anchors.py \
    --video .local/data/real/videos/pickle_vid_1_trimmed_from_8s.mp4 \
    --reference-annotations docs/annotations/real/anchors/annotations_v3.json \
    --n-samples 12 \
    --out docs/annotations/real/anchors/reprojection_labels_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720

DEFAULT_ANCHOR_KEYS = [
    "kitchen_near_left",
    "kitchen_near_right",
    "kitchen_far_left",
    "kitchen_far_right",
]

ANCHOR_LABELS = {
    "kitchen_near_left": "near-left kitchen corner",
    "kitchen_near_right": "near-right kitchen corner",
    "kitchen_far_left": "far-left kitchen corner",
    "kitchen_far_right": "far-right kitchen corner",
}


def _load_anchor_keys(reference_path: Path | None) -> list[str]:
    if reference_path is None or not reference_path.exists():
        return DEFAULT_ANCHOR_KEYS
    with open(reference_path) as f:
        data = json.load(f)
    frames = data.get("annotated_frames", [])
    if not frames:
        return DEFAULT_ANCHOR_KEYS
    anchors = frames[0].get("anchors", {})
    keys = [k for k in DEFAULT_ANCHOR_KEYS if k in anchors]
    return keys or list(anchors.keys())


def _read_existing(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    entries = data.get("frames", data.get("annotated_frames", []))
    out = {}
    for entry in entries:
        out[int(entry["frame_index"])] = entry.get("anchors", {})
    return out


def _display_scale(height: int, width: int) -> float:
    return min(MAX_DISPLAY_W / width, MAX_DISPLAY_H / height, 1.0)


def _parse_frames(args, total_frames: int) -> list[int]:
    if args.frames:
        return [int(x.strip()) for x in args.frames.split(",") if x.strip()]

    start = max(0, int(args.start_frame))
    end = total_frames - 1 if args.end_frame < 0 else min(total_frames - 1, int(args.end_frame))
    if end < start:
        raise ValueError("end_frame must be >= start_frame")
    n_samples = max(1, int(args.n_samples))
    return sorted({int(x) for x in np.linspace(start, end, n_samples)})


def _draw_ui(
    frame: np.ndarray,
    anchors: dict,
    anchor_keys: list[str],
    frame_idx: int,
    frame_pos: int,
    frame_total: int,
    scale: float,
) -> np.ndarray:
    out = cv2.resize(frame, None, fx=scale, fy=scale)

    for key in anchor_keys:
        if key not in anchors:
            continue
        x = int(round(anchors[key][0] * scale))
        y = int(round(anchors[key][1] * scale))
        cv2.circle(out, (x, y), 6, (0, 255, 255), -1)
        cv2.circle(out, (x, y), 6, (0, 0, 0), 1)
        cv2.putText(out, key, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, key, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    next_idx = len(anchors)
    next_key = anchor_keys[next_idx] if next_idx < len(anchor_keys) else "done"
    prompt = f"frame {frame_pos}/{frame_total}  idx={frame_idx}  next: {ANCHOR_LABELS.get(next_key, next_key)}"
    cv2.putText(out, prompt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(out, prompt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    help_text = "U undo   R reset frame   S save/next   N skip frame   Q save+quit"
    y = out.shape[0] - 12
    cv2.putText(out, help_text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(out, help_text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def _save_output(path: Path, video_path: Path, anchor_keys: list[str], saved: dict[int, dict]) -> None:
    payload = {
        "video": video_path.name,
        "anchor_keys": anchor_keys,
        "frames": [
            {"frame_index": idx, "anchors": saved[idx]}
            for idx in sorted(saved.keys())
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate sampled frames for reprojection validation")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--reference-annotations", default=None, help="Reference v3 anchor JSON")
    parser.add_argument("--frames", default=None, help="Comma-separated frame indices")
    parser.add_argument("--n-samples", type=int, default=12, help="Number of evenly spaced frames")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to sample")
    parser.add_argument("--end-frame", type=int, default=-1, help="Last frame to sample (-1 = end)")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)
    reference_path = Path(args.reference_annotations) if args.reference_annotations else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = _parse_frames(args, total_frames)
    anchor_keys = _load_anchor_keys(reference_path)
    saved = _read_existing(out_path)

    window_name = "annotate_reprojection_anchors"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for frame_pos, frame_idx in enumerate(frame_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Skipping unreadable frame {frame_idx}")
            continue

        scale = _display_scale(frame.shape[0], frame.shape[1])
        anchors = dict(saved.get(frame_idx, {}))

        click_state = {"anchors": anchors}

        def _on_mouse(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            current = click_state["anchors"]
            if len(current) >= len(anchor_keys):
                return
            key = anchor_keys[len(current)]
            current[key] = [x / scale, y / scale]

        cv2.setMouseCallback(window_name, _on_mouse)

        while True:
            canvas = _draw_ui(frame, click_state["anchors"], anchor_keys, frame_idx, frame_pos, len(frame_indices), scale)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("u"):
                for anchor_key in reversed(anchor_keys):
                    if anchor_key in click_state["anchors"]:
                        del click_state["anchors"][anchor_key]
                        break
            elif key == ord("r"):
                click_state["anchors"].clear()
            elif key == ord("n"):
                break
            elif key == ord("s"):
                if len(click_state["anchors"]) < len(anchor_keys):
                    print(f"Frame {frame_idx}: annotate all {len(anchor_keys)} anchors before saving")
                    continue
                saved[frame_idx] = dict(click_state["anchors"])
                _save_output(out_path, video_path, anchor_keys, saved)
                break
            elif key == ord("q"):
                if len(click_state["anchors"]) == len(anchor_keys):
                    saved[frame_idx] = dict(click_state["anchors"])
                _save_output(out_path, video_path, anchor_keys, saved)
                cap.release()
                cv2.destroyAllWindows()
                print(f"Saved {len(saved)} annotated frames to {out_path}")
                return

    cap.release()
    cv2.destroyAllWindows()
    _save_output(out_path, video_path, anchor_keys, saved)
    print(f"Saved {len(saved)} annotated frames to {out_path}")


if __name__ == "__main__":
    main()
