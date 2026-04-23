"""
Interactive annotation tool for pickleball court kitchen lines.

Camera setup: side/end-on view of the kitchen zone. Near baseline is
off-screen. Visible lines: near kitchen line (front), far kitchen line
(back), sidelines (slanted), net (center).

Click order
-----------
  1. Near kitchen line — LEFT end    [REQUIRED]
  2. Near kitchen line — RIGHT end   [REQUIRED]
  3. Far kitchen line  — LEFT end    [optional]
  4. Far kitchen line  — RIGHT end   [optional]

  The legal zone (green fill) is auto-derived from the near kitchen
  line — no extra click needed.

Keys
----
  P   toggle preview  (available after 2 clicks)
  U   undo last click
  R   reset all clicks
  S   save and quit   (any time after 2 clicks)
  Q   quit without saving

Usage
-----
  python scripts/annotate_anchors.py \\
      --video  data/real/videos/pickle_vid_1.MOV \\
      --frame  0 \\
      --out    data/real/annotations/annotations_v3.json
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

CLICK_ORDER = [
    ("kitchen_near_left",  "[1/4] Near kitchen line (front blue line) — LEFT end   [REQUIRED]"),
    ("kitchen_near_right", "[2/4] Near kitchen line (front blue line) — RIGHT end  [REQUIRED]"),
    ("kitchen_far_left",   "[3/4] Far kitchen line  (back blue line)  — LEFT end   [optional — S to skip]"),
    ("kitchen_far_right",  "[4/4] Far kitchen line  (back blue line)  — RIGHT end  [optional — S to skip]"),
]
N_REQUIRED = 2

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


def _derive_legal_ref(near_l: tuple, near_r: tuple) -> list:
    """
    Auto-generate a point on the camera side of the near kitchen line.
    Uses the midpoint shifted 80px downward in image coords (y increases
    toward camera for a side/end-on mount).
    """
    mx = (near_l[0] + near_r[0]) / 2.0
    my = (near_l[1] + near_r[1]) / 2.0
    return [mx, my + 80.0]


def _build_anchors(clicks: list, scale: float = 1.0) -> dict:
    """Build anchor dict from clicks, auto-adding legal_ref_near."""
    anchors = {}
    for i, pt in enumerate(clicks):
        key = CLICK_ORDER[i][0]
        anchors[key] = [float(pt[0]) * scale, float(pt[1]) * scale]
    # Auto-derive legal reference from the near kitchen line
    if "kitchen_near_left" in anchors and "kitchen_near_right" in anchors:
        nl = anchors["kitchen_near_left"]
        nr = anchors["kitchen_near_right"]
        ref = _derive_legal_ref(nl, nr)
        anchors["legal_ref_near"] = ref
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

    # Draw completed anchor dots
    for i, (fx, fy) in enumerate(clicks):
        key, _ = CLICK_ORDER[i]
        dx, dy = int(fx * sc), int(fy * sc)
        color = COLOR_REQUIRED if i < N_REQUIRED else COLOR_OPTIONAL
        cv2.circle(out, (dx, dy), 7, color, -1)
        cv2.circle(out, (dx, dy), 7, (0, 0, 0), 1)
        label = "near-L" if key == "kitchen_near_left" else \
                "near-R" if key == "kitchen_near_right" else \
                "far-L"  if key == "kitchen_far_left"  else "far-R"
        cv2.putText(out, label, (dx + 9, dy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Status bar
    idx = len(clicks)
    if idx < len(CLICK_ORDER):
        desc = CLICK_ORDER[idx][1]
    else:
        desc = "All done — press S to save"
    cv2.putText(out, desc, (10, H_d - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
    cv2.putText(out, desc, (10, H_d - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_TEXT, 1)

    hint = "P=preview  U=undo  R=reset  S=save  Q=quit"
    cv2.putText(out, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
    cv2.putText(out, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT, 1)

    return out


def main(video_path: Path, frame_idx: int, out_path: Path) -> None:
    frame = _read_frame(video_path, frame_idx)
    H_src, W_src = frame.shape[:2]
    sc = _scale(H_src, W_src)

    clicks: list[tuple[int, int]] = []
    preview = False

    win = "Annotate Kitchen Lines"
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

    anchors = _build_anchors(clicks)  # full-resolution, includes auto legal_ref

    annotation = {
        "_notes": (
            f"Kitchen line anchor points ({W_src}x{H_src} px, frame {frame_idx}). "
            "Camera is side/end-on — near baseline is off-screen. "
            "legal_ref_near is auto-derived (not manually clicked). "
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
        print(f"  Near kitchen: {[round(c) for c in kp['near'][0]]} → {[round(c) for c in kp['near'][1]]}")
        if "far" in kp:
            print(f"  Far  kitchen: {[round(c) for c in kp['far'][0]]} → {[round(c) for c in kp['far'][1]]}")
        print(f"  Legal sign: {m.legal_near_sign()}")
    except Exception as e:
        print(f"  Geometry check: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default="data/real/annotations/annotations_v3.json")
    args = parser.parse_args()
    main(Path(args.video), args.frame, Path(args.out))
