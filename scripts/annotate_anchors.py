"""
Interactive anchor-point annotation tool for pickleball court registration (v3).

Click points in this order:
  1. near_left          — bottom-left corner of pickleball court (near camera)
  2. near_right         — bottom-right corner (near camera)
  3. net_left           — left end of net (where net meets left sideline)
  4. net_right          — right end of net
  5. far_left           — far-left corner of court (behind net)
  6. far_right          — far-right corner of court
  7. kitchen_near_left  — near kitchen line × left sideline  (optional)
  8. kitchen_near_right — near kitchen line × right sideline (optional)
  9. kitchen_far_left   — far kitchen line × left sideline   (optional)
 10. kitchen_far_right  — far kitchen line × right sideline  (optional)
 11. legal_ref_near     — a point clearly in the near legal zone

Keys
----
  U   undo last click
  R   reset all clicks
  P   preview inferred court geometry (requires 6 required anchors)
  S   save and quit
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
from src.viz import draw_court_model, draw_frame_info

CLICK_ORDER = [
    ("near_left",          "near-left corner (near camera, left side)"),
    ("near_right",         "near-right corner (near camera, right side)"),
    ("net_left",           "left end of net"),
    ("net_right",          "right end of net"),
    ("far_left",           "far-left corner (behind net)"),
    ("far_right",          "far-right corner (behind net)"),
    ("kitchen_near_left",  "near kitchen line × left sideline  [OPTIONAL — press S to skip]"),
    ("kitchen_near_right", "near kitchen line × right sideline [OPTIONAL — press S to skip]"),
    ("kitchen_far_left",   "far kitchen line × left sideline   [OPTIONAL — press S to skip]"),
    ("kitchen_far_right",  "far kitchen line × right sideline  [OPTIONAL — press S to skip]"),
    ("legal_ref_near",     "a point clearly in the near legal zone (behind near kitchen line)"),
]
N_REQUIRED = 6  # first 6 are required

MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def _display_scale(H: int, W: int) -> float:
    scale = min(MAX_DISPLAY_W / W, MAX_DISPLAY_H / H, 1.0)
    return scale


def _render(base: np.ndarray, clicks: list, scale: float, preview: bool) -> np.ndarray:
    out = cv2.resize(base, None, fx=scale, fy=scale)
    H_d, W_d = out.shape[:2]

    # Draw completed clicks
    for i, (x, y) in enumerate(clicks):
        key, desc = CLICK_ORDER[i]
        dx, dy = int(x * scale), int(y * scale)
        color = (0, 255, 0) if i < N_REQUIRED else (0, 200, 255)
        cv2.circle(out, (dx, dy), 6, color, -1)
        cv2.circle(out, (dx, dy), 6, (0, 0, 0), 1)
        cv2.putText(out, key, (dx + 8, dy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Show next expected click
    idx = len(clicks)
    if idx < len(CLICK_ORDER):
        key, desc = CLICK_ORDER[idx]
        prompt = f"[{idx + 1}/{len(CLICK_ORDER)}]  {desc}"
        cv2.putText(out, prompt, (10, H_d - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, prompt, (10, H_d - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Preview inferred geometry
    if preview and len(clicks) >= N_REQUIRED:
        anchor_names = [CLICK_ORDER[i][0] for i in range(len(clicks))]
        anchors = dict(zip(anchor_names, clicks))
        try:
            model = CourtGeometryModel(anchors)
            # Scale anchors for display
            scaled_anchors = {k: [v[0] * scale, v[1] * scale]
                              for k, v in anchors.items()}
            scaled_model = CourtGeometryModel(scaled_anchors)
            legal_ref = clicks[10][0] * scale, clicks[10][1] * scale if len(clicks) > 10 else (
                scaled_anchors.get("legal_ref_near", [W_d // 2, H_d * 0.8])
            )
            sign = scaled_model.legal_near_sign(
                tuple(scaled_anchors.get("legal_ref_near",
                                         [W_d // 2, H_d * 0.8]))
            )
            out = draw_court_model(out, scaled_model, legal_sign=sign,
                                   draw_anchors=False, fallback=False)
        except Exception:
            pass

    # Instructions
    cv2.putText(out, "U=undo  R=reset  P=preview  S=save  Q=quit",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(out, "U=undo  R=reset  P=preview  S=save  Q=quit",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return out


def main(video_path: Path, frame_idx: int, out_path: Path) -> None:
    frame = _read_frame(video_path, frame_idx)
    H_src, W_src = frame.shape[:2]
    scale = _display_scale(H_src, W_src)

    clicks: list = []  # list of (x, y) in full-res coords
    preview = False

    win = "Annotate Court Anchors"
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

        if key == ord("u") or key == ord("U"):
            if clicks:
                clicks.pop()
        elif key == ord("r") or key == ord("R"):
            clicks.clear()
            preview = False
        elif key == ord("p") or key == ord("P"):
            preview = not preview
        elif key == ord("s") or key == ord("S"):
            if len(clicks) < N_REQUIRED:
                print(f"Need at least {N_REQUIRED} clicks before saving.")
                continue
            break
        elif key == ord("q") or key == ord("Q"):
            print("Quit without saving.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # Build anchors dict from clicks
    anchor_keys = [CLICK_ORDER[i][0] for i in range(len(clicks))]
    anchors = {k: list(v) for k, v in zip(anchor_keys, clicks)}

    annotation = {
        "_notes": (
            "Pickleball court anchor points in full-resolution pixel coordinates "
            f"({W_src}x{H_src}). Annotated on frame {frame_idx}. "
            "Required anchors: near_left, near_right, far_left, far_right, net_left, net_right. "
            "Kitchen-line anchors override proportional inference (7/22 from net). "
            "Run scripts/annotate_anchors.py to re-annotate."
        ),
        "video": video_path.name,
        "reference_frame_index": frame_idx,
        "annotated_frames": [
            {
                "frame_index": frame_idx,
                "anchors": anchors,
            }
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(annotation, f, indent=2)
    print(f"Saved: {out_path}")

    # Validate
    try:
        model = CourtGeometryModel(anchors)
        kp = model.kitchen_endpoints()
        print(f"  Near kitchen: {kp['near']}")
        print(f"  Far kitchen:  {kp['far']}")
    except Exception as e:
        print(f"  Warning: geometry validation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate pickleball court anchors (v3)")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to annotate")
    parser.add_argument(
        "--out",
        default="data/real/annotations/annotations_v3.json",
        help="Output JSON path",
    )
    args = parser.parse_args()
    main(Path(args.video), args.frame, Path(args.out))
