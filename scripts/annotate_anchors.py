"""
Interactive anchor-point annotation tool for pickleball court registration (v3).

Click the court anchors in the order shown in the status bar.
The first 5 clicks are REQUIRED; 6-9 are OPTIONAL (press S to skip at any time).

Click order
-----------
  1. kitchen_near_left    Left end of the FRONT blue kitchen/NVZ line  [REQUIRED]
  2. kitchen_near_right   Right end of the FRONT blue kitchen/NVZ line [REQUIRED]
  3. near_left            Bottom-left corner of the pickleball court    [REQUIRED]
  4. near_right           Bottom-right corner of the pickleball court   [REQUIRED]
  5. legal_ref_near       Any point BEHIND the kitchen line (between    [REQUIRED]
                          kitchen line and near baseline — the legal zone)
  6. kitchen_far_left     Left end of the BACK blue kitchen/NVZ line    [optional]
  7. kitchen_far_right    Right end of the BACK blue kitchen/NVZ line   [optional]
  8. far_left             Top-left court boundary corner                 [optional]
  9. far_right            Top-right court boundary corner                [optional]

Keys
----
  P   preview inferred court geometry (active after 5 clicks)
  U   undo last click
  R   reset all
  S   save and quit (any time after 5 clicks)
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
    ("kitchen_near_left",  "[1/9] FRONT blue kitchen line — LEFT end  [REQUIRED]"),
    ("kitchen_near_right", "[2/9] FRONT blue kitchen line — RIGHT end [REQUIRED]"),
    ("near_left",          "[3/9] Near baseline — LEFT corner         [REQUIRED]"),
    ("near_right",         "[4/9] Near baseline — RIGHT corner        [REQUIRED]"),
    ("legal_ref_near",     "[5/9] Any point BEHIND the kitchen line   [REQUIRED]"),
    ("kitchen_far_left",   "[6/9] BACK blue kitchen line — LEFT end   [optional — S to skip]"),
    ("kitchen_far_right",  "[7/9] BACK blue kitchen line — RIGHT end  [optional — S to skip]"),
    ("far_left",           "[8/9] Far baseline — LEFT corner          [optional — S to skip]"),
    ("far_right",          "[9/9] Far baseline — RIGHT corner         [optional — S to skip]"),
]
N_REQUIRED = 5

MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720

# Colors (BGR)
COLOR_REQUIRED = (0, 255, 80)
COLOR_OPTIONAL = (0, 200, 255)
COLOR_TEXT     = (255, 255, 255)


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def _display_scale(H: int, W: int) -> float:
    return min(MAX_DISPLAY_W / W, MAX_DISPLAY_H / H, 1.0)


def _try_preview(clicks: list, scale: float, frame: np.ndarray) -> np.ndarray:
    """Overlay inferred court geometry on frame. Returns frame unchanged on error."""
    anchor_keys = [CLICK_ORDER[i][0] for i in range(len(clicks))]
    anchors_fullres = dict(zip(anchor_keys, [list(c) for c in clicks]))
    # Scale anchors to display resolution
    anchors_disp = {k: [v[0] * scale, v[1] * scale] for k, v in anchors_fullres.items()}
    try:
        model = CourtGeometryModel(anchors_disp)
        sign = model.legal_near_sign()
        return draw_court_model(frame, model, legal_sign=sign,
                                draw_anchors=False, fallback=False)
    except Exception:
        return frame


def _render(
    base: np.ndarray,
    clicks: list,
    scale: float,
    preview: bool,
) -> np.ndarray:
    out = cv2.resize(base, None, fx=scale, fy=scale)
    H_d, W_d = out.shape[:2]

    if preview and len(clicks) >= N_REQUIRED:
        out = _try_preview(clicks, scale, out)

    # Dot + label for each completed click
    for i, (fx, fy) in enumerate(clicks):
        key, _ = CLICK_ORDER[i]
        dx, dy = int(fx * scale), int(fy * scale)
        color = COLOR_REQUIRED if i < N_REQUIRED else COLOR_OPTIONAL
        cv2.circle(out, (dx, dy), 7, color, -1)
        cv2.circle(out, (dx, dy), 7, (0, 0, 0), 1)
        cv2.putText(out, key, (dx + 9, dy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # Status bar — next expected click
    idx = len(clicks)
    if idx < len(CLICK_ORDER):
        _, desc = CLICK_ORDER[idx]
        cv2.putText(out, desc, (10, H_d - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, desc, (10, H_d - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
    else:
        done = "All done — press S to save"
        cv2.putText(out, done, (10, H_d - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, done, (10, H_d - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)

    # Key hints
    cv2.putText(out, "P=preview  U=undo  R=reset  S=save  Q=quit",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(out, "P=preview  U=undo  R=reset  S=save  Q=quit",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    return out


def main(video_path: Path, frame_idx: int, out_path: Path) -> None:
    frame = _read_frame(video_path, frame_idx)
    H_src, W_src = frame.shape[:2]
    scale = _display_scale(H_src, W_src)

    clicks: list[tuple[int, int]] = []
    preview = False

    win = "Annotate Court Anchors — v3"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(CLICK_ORDER):
            fx = int(round(x / scale))
            fy = int(round(y / scale))
            clicks.append((fx, fy))

    cv2.setMouseCallback(win, on_click)

    while True:
        disp = _render(frame, clicks, scale, preview)
        cv2.imshow(win, disp)
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
                continue
            break
        elif key in (ord("q"), ord("Q")):
            print("Quit without saving.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # Build anchor dict
    anchor_keys = [CLICK_ORDER[i][0] for i in range(len(clicks))]
    anchors = {k: list(v) for k, v in zip(anchor_keys, clicks)}

    annotation = {
        "_notes": (
            f"Pickleball court anchor points ({W_src}x{H_src} px). "
            f"Annotated on frame {frame_idx}. "
            "Required: kitchen_near_left/right, near_left/right, legal_ref_near. "
            "Net and far corners are inferred from kitchen proportions if absent. "
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

    # Quick validation printout
    try:
        model = CourtGeometryModel(anchors)
        kp = model.kitchen_endpoints()
        print(f"  Near kitchen: {[round(c) for c in kp['near'][0]]} → {[round(c) for c in kp['near'][1]]}")
        print(f"  Far  kitchen: {[round(c) for c in kp['far'][0]]} → {[round(c) for c in kp['far'][1]]}")
        print(f"  Legal sign:   {model.legal_near_sign()}")
    except Exception as e:
        print(f"  Geometry check failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate pickleball court anchors (v3)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default="data/real/annotations/annotations_v3.json")
    args = parser.parse_args()
    main(Path(args.video), args.frame, Path(args.out))
