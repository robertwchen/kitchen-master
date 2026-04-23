"""
Pickleball court geometry model for a side/end-on camera showing the kitchen zone.

The camera is positioned just outside the kitchen area and does not show the
near baseline. Visible structures: near kitchen/NVZ line, net, far kitchen line,
sidelines (slanted).

Primary inputs are the directly annotated kitchen line endpoints (clicked on the
visible blue lines). Everything else is derived or optional.
"""

import cv2
import numpy as np
from typing import Optional

from src.court_registration import LineModel


class CourtGeometryModel:
    """
    Court geometry from directly annotated kitchen line endpoints.

    Required anchors
    ----------------
    kitchen_near_left    Left end of the near (front) blue NVZ line.
    kitchen_near_right   Right end of the near (front) blue NVZ line.
    legal_ref_near       Any point on the legal side of the near kitchen line
                         (between kitchen line and camera — below the line in image).

    Optional anchors
    ----------------
    kitchen_far_left     Left end of the far (back) blue NVZ line.
    kitchen_far_right    Right end of the far (back) blue NVZ line.
    net_left             Left end of the net (where net meets left sideline).
    net_right            Right end of the net.
    sideline_near_left   A point on the left sideline (near side).
    sideline_near_right  A point on the right sideline (near side).
    """

    REQUIRED = {
        "kitchen_near_left",
        "kitchen_near_right",
        "legal_ref_near",
    }

    def __init__(self, anchors: dict) -> None:
        missing = self.REQUIRED - set(anchors.keys())
        if missing:
            raise ValueError(f"Missing required anchors: {sorted(missing)}")

        self._raw: dict[str, np.ndarray] = {
            k: np.array(v, dtype=float) for k, v in anchors.items()
        }
        self._build_geometry()

    def _build_geometry(self) -> None:
        r = self._raw
        self._kn_l = r["kitchen_near_left"]
        self._kn_r = r["kitchen_near_right"]

        self.near_kitchen_line = LineModel(
            tuple(self._kn_l), tuple(self._kn_r)
        )

        # Far kitchen line (optional)
        if "kitchen_far_left" in r and "kitchen_far_right" in r:
            self._kf_l = r["kitchen_far_left"]
            self._kf_r = r["kitchen_far_right"]
            self.far_kitchen_line: Optional[LineModel] = LineModel(
                tuple(self._kf_l), tuple(self._kf_r)
            )
        else:
            self._kf_l = self._kf_r = None
            self.far_kitchen_line = None

        # Net line (optional)
        if "net_left" in r and "net_right" in r:
            self.net_line: Optional[LineModel] = LineModel(
                tuple(r["net_left"]), tuple(r["net_right"])
            )
        else:
            self.net_line = None

        # Legal zone polygon: covers the entire region on the legal (camera) side
        # of the near kitchen line, reaching all the way to the frame edges.
        #
        # Two extensions are needed:
        #   1. Lateral (along the line): push the endpoints far left/right so
        #      the polygon spans the full frame width regardless of where the
        #      user clicked.
        #   2. Perpendicular (toward camera): push those extended corners far in
        #      the legal direction so the polygon reaches the frame bottom/sides.
        kn_l, kn_r = self._kn_l, self._kn_r
        LATERAL = 5000.0   # px along the kitchen line direction
        PERP = 5000.0      # px toward camera (perpendicular to line)

        # Unit vector along the line (left → right)
        line_dir = kn_r - kn_l
        line_len = np.linalg.norm(line_dir)
        if line_len > 1e-9:
            line_dir = line_dir / line_len

        # Extrapolate the line far past both endpoints
        far_l = kn_l - line_dir * LATERAL
        far_r = kn_r + line_dir * LATERAL

        # Unit normal pointing toward the legal (camera) side
        sign = self.legal_near_sign()
        na = self.near_kitchen_line.a * sign
        nb = self.near_kitchen_line.b * sign
        nn = np.sqrt(na * na + nb * nb)
        if nn > 1e-9:
            na /= nn
            nb /= nn
        perp = np.array([na * PERP, nb * PERP])

        self.near_legal_polygon = np.array(
            [far_l, far_r, far_r + perp, far_l + perp], dtype=np.float32
        )

        # Far legal polygon (opposite side of far kitchen line)
        if self._kf_l is not None and self._kf_r is not None:
            kf_l, kf_r = self._kf_l, self._kf_r
            far_kf_dir = kf_r - kf_l
            far_kf_len = np.linalg.norm(far_kf_dir)
            if far_kf_len > 1e-9:
                far_kf_dir = far_kf_dir / far_kf_len
            ffar_l = kf_l - far_kf_dir * LATERAL
            ffar_r = kf_r + far_kf_dir * LATERAL

            fa = self.far_kitchen_line.a * (-sign)
            fb = self.far_kitchen_line.b * (-sign)
            fn = np.sqrt(fa * fa + fb * fb)
            if fn > 1e-9:
                fa /= fn
                fb /= fn
            fperp = np.array([fa * PERP, fb * PERP])

            self.far_legal_polygon: Optional[np.ndarray] = np.array(
                [ffar_l, ffar_r, ffar_r + fperp, ffar_l + fperp], dtype=np.float32
            )
        else:
            self.far_legal_polygon = None

    # ── public API ────────────────────────────────────────────────────────────

    def legal_near_sign(self, ref_pt: Optional[tuple] = None) -> int:
        """Return +1/-1 indicating which side of the near kitchen line is legal."""
        if ref_pt is None:
            ref_pt = tuple(self._raw["legal_ref_near"].tolist())
        d = self.near_kitchen_line.signed_distance(ref_pt)
        return 1 if d >= 0 else -1

    def anchor_dict(self) -> dict:
        return {k: v.tolist() for k, v in self._raw.items()}

    def kitchen_endpoints(self) -> dict:
        out = {
            "near": (tuple(self._kn_l.tolist()), tuple(self._kn_r.tolist())),
        }
        if self._kf_l is not None:
            out["far"] = (tuple(self._kf_l.tolist()), tuple(self._kf_r.tolist()))
        return out

    def warp(self, H: np.ndarray) -> "CourtGeometryModel":
        """Return a new model with every anchor point warped through homography H."""
        new_anchors: dict = {}
        for key, pt in self._raw.items():
            p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(p, H)
            new_anchors[key] = [float(warped[0, 0, 0]), float(warped[0, 0, 1])]
        return CourtGeometryModel(new_anchors)
