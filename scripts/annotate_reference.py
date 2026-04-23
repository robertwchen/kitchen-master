"""
Interactive click-to-annotate tool for marking kitchen line endpoints.

Click order:
    Clicks 1–2  → near kitchen line endpoints (p1, p2)
    Clicks 3–4  → far kitchen line endpoints  (p1, p2)
    Click  5    → legal-side reference point

Keys:
    u  — undo last click
    s  — save and exit
    q  — quit without saving
    r  — reset all clicks

Usage:
    python scripts/annotate_reference.py \\
        --video data/real/videos/pickle_vid_1.MOV \\
        --frame 60 \\
        --out   data/real/annotations/annotations.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

COLORS = {
    "near": (0, 220, 0),    # green
    "far":  (0, 180, 255),  # orange
    "ref":  (200, 0, 200),  # magenta
}

LABELS = [
    ("near_kitchen_line", "p1",   "NEAR kitchen line — click endpoint 1"),
    ("near_kitchen_line", "p2",   "NEAR kitchen line — click endpoint 2"),
    ("far_kitchen_line",  "p1",   "FAR  kitchen line — click endpoint 1"),
    ("far_kitchen_line",  "p2",   "FAR  kitchen line — click endpoint 2"),
    ("legal_side",        "ref",  "LEGAL-SIDE reference point (any point behind the near line)"),
]


def _load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: could not read frame {frame_idx} from {video_path}")
        sys.exit(1)
    return frame


def _draw_state(base: np.ndarray, clicks: list[tuple[int, int]]) -> np.ndarray:
    canvas = base.copy()
    for i, (x, y) in enumerate(clicks):
        key, pt, _ = LABELS[i]
        color = COLORS["near"] if "near" in key else COLORS["far"] if "far" in key else COLORS["ref"]
        cv2.circle(canvas, (x, y), 8, color, -1)
        cv2.circle(canvas, (x, y), 8, (255, 255, 255), 2)
        cv2.putText(canvas, f"{key[:4]}.{pt}", (x + 10, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Draw lines when both endpoints are placed
    if len(clicks) >= 2:
        cv2.line(canvas, clicks[0], clicks[1], COLORS["near"], 2)
    if len(clicks) >= 4:
        cv2.line(canvas, clicks[2], clicks[3], COLORS["far"], 2)

    # Status prompt
    if len(clicks) < len(LABELS):
        prompt = LABELS[len(clicks)][2]
    else:
        prompt = "Done — press S to save, Q to quit"
    cv2.putText(canvas, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "U=undo  R=reset  S=save  Q=quit", (10, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return canvas


def run_annotation(video_path: Path, frame_idx: int, out_path: Path) -> None:
    base = _load_frame(video_path, frame_idx)
    # Downscale for display if needed
    H, W = base.shape[:2]
    scale = min(1.0, 1280 / W, 800 / H)
    display_base = cv2.resize(base, (int(W * scale), int(H * scale))) if scale < 1 else base

    clicks: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(LABELS):
            # Scale back to full-resolution coordinates
            clicks.append((int(x / scale), int(y / scale)))

    win = "KitchenMaster — annotate reference frame"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        # Downscale display frame
        display_clicks = [(int(x * scale), int(y * scale)) for x, y in clicks]
        canvas = _draw_state(display_base, display_clicks)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u") and clicks:
            clicks.pop()
        elif key == ord("r"):
            clicks.clear()
        elif key == ord("s"):
            if len(clicks) < 4:
                print("Need at least 4 clicks (2 per kitchen line) before saving.")
            else:
                break
        elif key == ord("q"):
            print("Quit without saving.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    frame_data: dict = {"frame_index": frame_idx}
    for i, (key, pt, _) in enumerate(LABELS[:4]):
        frame_data.setdefault(key, {})[pt] = list(clicks[i])
    if len(clicks) >= 5:
        frame_data["legal_side_reference_point"] = list(clicks[4])

    annotation = {
        "_notes": (
            "Pixel coordinates (x, y) from top-left of the full-resolution frame. "
            "legal_side_reference_point is any point known to be on the legal "
            "(non-fault) side of the near kitchen line."
        ),
        "video": video_path.name,
        "annotated_frames": [frame_data],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(annotation, f, indent=2)
    print(f"Saved annotation to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Click-to-annotate kitchen line reference")
    parser.add_argument("--video", default="data/real/videos/pickle_vid_1.MOV")
    parser.add_argument("--frame", type=int, default=60, help="Frame index to annotate")
    parser.add_argument("--out", default="data/real/annotations/annotations.json")
    args = parser.parse_args()

    run_annotation(Path(args.video), args.frame, Path(args.out))
