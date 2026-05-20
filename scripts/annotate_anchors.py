"""
Interactive annotation tool for pickleball NVZ boundaries (side-facing camera).

Camera setup: camera faces the court from the side. The kitchen zone appears
as a rectangle. The two NVZ boundary lines you see from this angle are the
LEFT edge and RIGHT edge of that rectangle:

    far-L ──────────────── far-R        ← back edge (for reference)
      |                      |
  LEFT NVZ               RIGHT NVZ
  BOUNDARY               BOUNDARY
      |                      |
   near-L ──────────────── near-R       ← front edge (for reference)

Click the four corners of the kitchen rectangle.  Clicks 1+3 define the
LEFT NVZ boundary line; clicks 2+4 define the RIGHT NVZ boundary line.
The green legal zones (outside both boundaries) appear in the preview
once all four corners are placed.

Click order
-----------
  1. LEFT  NVZ boundary — NEAR end   (front-left  corner)  [REQUIRED]
  2. RIGHT NVZ boundary — NEAR end   (front-right corner)  [REQUIRED]
  3. LEFT  NVZ boundary — FAR  end   (back-left   corner)  [optional]
  4. RIGHT NVZ boundary — FAR  end   (back-right  corner)  [optional]

Keys
----
  P   toggle preview  (green zones visible after all 4 clicks)
  U   undo last click
  R   reset all clicks
  S   save and quit   (any time after 2 clicks)
  Q   quit without saving

Usage
-----
  python scripts/annotate_anchors.py \\
      --video  .local/data/real/videos/pickle_vid_1.MOV \\
      --frame  0 \\
      --out    docs/annotations/real/anchors/annotations_v3.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.court_model import CourtGeometryModel
from src.viz import draw_court_model

# clicks 1+3 → LEFT NVZ boundary line
# clicks 2+4 → RIGHT NVZ boundary line
CLICK_ORDER = [
    ("kitchen_near_left",  "[1/4] LEFT NVZ boundary  — NEAR end  (front-left corner)   [REQUIRED]"),
    ("kitchen_near_right", "[2/4] RIGHT NVZ boundary — NEAR end  (front-right corner)  [REQUIRED]"),
    ("kitchen_far_left",   "[3/4] LEFT NVZ boundary  — FAR end   (back-left corner)    [optional — S to skip]"),
    ("kitchen_far_right",  "[4/4] RIGHT NVZ boundary — FAR end   (back-right corner)   [optional — S to skip]"),
]
N_REQUIRED = 2

# Dot labels shown in the preview image
DOT_LABELS = {
    "kitchen_near_left":  "L-near",
    "kitchen_near_right": "R-near",
    "kitchen_far_left":   "L-far",
    "kitchen_far_right":  "R-far",
}

MAX_W, MAX_H = 1280, 720
COLOR_REQUIRED = (0, 255, 80)
COLOR_OPTIONAL = (0, 200, 255)
COLOR_TEXT     = (255, 255, 255)


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx}")
    return frame


def _scale(H: int, W: int) -> float:
    return min(MAX_W / W, MAX_H / H, 1.0)


def _build_anchors(clicks: list, scale: float = 1.0) -> dict:
    """Build anchor dict from click list at the given pixel scale."""
    anchors = {}
    for i, pt in enumerate(clicks):
        key = CLICK_ORDER[i][0]
        anchors[key] = [float(pt[0]) * scale, float(pt[1]) * scale]
    return anchors


def _try_preview(clicks: list, scale: float, disp: np.ndarray) -> np.ndarray:
    anchors = _build_anchors(clicks, scale)
    try:
        model = CourtGeometryModel(anchors)
        return draw_court_model(disp, model, draw_anchors=False)
    except Exception:
        return disp


def _render(base: np.ndarray, clicks: list, sc: float, preview: bool) -> np.ndarray:
    out = cv2.resize(base, None, fx=sc, fy=sc)
    H_d, W_d = out.shape[:2]

    if preview and len(clicks) >= N_REQUIRED:
        out = _try_preview(clicks, sc, out)

    # Anchor dots
    for i, (fx, fy) in enumerate(clicks):
        key, _ = CLICK_ORDER[i]
        dx, dy = int(fx * sc), int(fy * sc)
        color = COLOR_REQUIRED if i < N_REQUIRED else COLOR_OPTIONAL
        cv2.circle(out, (dx, dy), 7, color, -1)
        cv2.circle(out, (dx, dy), 7, (0, 0, 0), 1)
        cv2.putText(out, DOT_LABELS[key], (dx + 9, dy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # Status bar — next click description
    idx = len(clicks)
    if idx < len(CLICK_ORDER):
        desc = CLICK_ORDER[idx][1]
    elif idx < 4:
        desc = "Clicks 1+3 = LEFT NVZ line  |  Clicks 2+4 = RIGHT NVZ line  |  S to save"
    else:
        desc = "All 4 corners set — P for preview  |  S to save"
    cv2.putText(out, desc, (10, H_d - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
    cv2.putText(out, desc, (10, H_d - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_TEXT, 1)

    # Key hints
    hint = "P=preview(needs 4 clicks)  U=undo  R=reset  S=save  Q=quit"
    cv2.putText(out, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3)
    cv2.putText(out, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_TEXT, 1)

    return out


def main(video_path: Path, frame_idx: int, out_path: Path) -> None:
    frame = _read_frame(video_path, frame_idx)
    H_src, W_src = frame.shape[:2]
    sc = _scale(H_src, W_src)

    clicks: list[tuple[int, int]] = []
    preview = False

    win = "Annotate NVZ Boundaries (side-facing camera)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(CLICK_ORDER):
            clicks.append((int(round(x / sc)), int(round(y / sc))))

    cv2.setMouseCallback(win, on_click)

    while True:
        cv2.imshow(win, _render(frame, clicks, sc, preview))
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("u"), ord("U")):
            if clicks:
                clicks.pop()
        elif key in (ord("r"), ord("R")):
            clicks.clear()
            preview = False
        elif key in (ord("p"), ord("P")):
            if len(clicks) >= N_REQUIRED:
                preview = not preview
            else:
                print(f"Need at least {N_REQUIRED} clicks to preview.")
        elif key in (ord("s"), ord("S")):
            if len(clicks) < N_REQUIRED:
                print(f"Need at least {N_REQUIRED} clicks before saving.")
            else:
                break
        elif key in (ord("q"), ord("Q")):
            print("Quit without saving.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    anchors = _build_anchors(clicks)

    annotation = {
        "_notes": (
            f"NVZ boundary anchor points ({W_src}x{H_src} px, frame {frame_idx}). "
            "Side-facing camera — near baseline off-screen. "
            "LEFT NVZ line = kitchen_near_left → kitchen_far_left. "
            "RIGHT NVZ line = kitchen_near_right → kitchen_far_right. "
            "Re-annotate with scripts/annotate_anchors.py if overlay drifts."
        ),
        "video": video_path.name,
        "reference_frame_index": frame_idx,
        "annotated_frames": [{"frame_index": frame_idx, "anchors": anchors}],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(annotation, f, indent=2)
    print(f"Saved {len(clicks)} anchors → {out_path}")

    try:
        m = CourtGeometryModel(anchors)
        kp = m.kitchen_endpoints()
        print(f"  Near edge: {[round(c) for c in kp['near'][0]]} → {[round(c) for c in kp['near'][1]]}")
        if "far" in kp:
            print(f"  Far  edge: {[round(c) for c in kp['far'][0]]} → {[round(c) for c in kp['far'][1]]}")
            print(f"  LEFT  NVZ line: near-L {[round(c) for c in kp['near'][0]]} → far-L {[round(c) for c in kp['far'][0]]}")
            print(f"  RIGHT NVZ line: near-R {[round(c) for c in kp['near'][1]]} → far-R {[round(c) for c in kp['far'][1]]}")
            print(f"  Left  legal polygon: {m.left_legal_polygon is not None}")
            print(f"  Right legal polygon: {m.right_legal_polygon is not None}")
    except Exception as e:
        print(f"  Geometry check: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default="docs/annotations/real/anchors/annotations_v3.json")
    args = parser.parse_args()
    main(Path(args.video), args.frame, Path(args.out))
