"""
Pickleball court geometry model derived from anchor points.

Primary inputs are the directly annotated kitchen/NVZ lines (which are
visible as blue lines on the court), plus the near-side baseline corners.
The net position and far-side geometry are inferred from these.

All derived geometry (net, far kitchen, legal zones, court polygon) is
computed from the primary anchors. CourtGeometryModel.warp(H) propagates
the entire model through a homography for per-frame registration.
"""

import cv2
import numpy as np
from typing import Optional

from src.court_registration import LineModel

# NVZ (kitchen) line is 7 ft from net.  Half-court depth = 22 ft.
KITCHEN_FRAC: float = 7.0 / 22.0
# Inverse: given kitchen line + near baseline, net = kitchen + (7/15)*(kitchen - baseline)
NET_FROM_KITCHEN_FRAC: float = 7.0 / 15.0   # = KITCHEN_FRAC / (1 - KITCHEN_FRAC)


class CourtGeometryModel:
    """
    Pickleball court geometry anchored in pixel coordinates.

    Required anchors
    ----------------
    kitchen_near_left    Left end of the near (front) kitchen/NVZ line.
    kitchen_near_right   Right end of the near (front) kitchen/NVZ line.
    near_left            Near-side baseline, left court corner.
    near_right           Near-side baseline, right court corner.
    legal_ref_near       Any point clearly BEHIND the near kitchen line
                         (in the legal zone, i.e. between kitchen and near baseline).

    Optional anchors (improve geometry accuracy if visible)
    -------------------------------------------------------
    kitchen_far_left     Left end of the far (back) kitchen/NVZ line.
    kitchen_far_right    Right end of the far (back) kitchen/NVZ line.
    far_left             Far-side baseline, left corner.
    far_right            Far-side baseline, right corner.
    net_left             Left sideline endpoint of the net.
    net_right            Right sideline endpoint of the net.

    Derived geometry (auto-filled when optional anchors are absent)
    ---------------------------------------------------------------
    net      Inferred at 7/15 past the kitchen line away from baseline.
    far      Inferred by reflecting near corners through net.
    far kitchen   Inferred at 7/22 from net toward far baseline.
    """

    REQUIRED = {
        "kitchen_near_left", "kitchen_near_right",
        "near_left", "near_right",
        "legal_ref_near",
    }

    def __init__(self, anchors: dict) -> None:
        missing = self.REQUIRED - set(anchors.keys())
        if missing:
            raise ValueError(f"Missing required anchors: {sorted(missing)}")

        self._raw: dict[str, np.ndarray] = {
            k: np.array(v, dtype=float) for k, v in anchors.items()
        }
        self._fill_derived()
        self._build_geometry()

    # ── derivation ────────────────────────────────────────────────────────────

    def _fill_derived(self) -> None:
        r = self._raw
        kn_l = r["kitchen_near_left"]
        kn_r = r["kitchen_near_right"]
        near_l = r["near_left"]
        near_r = r["near_right"]

        # Net: placed at 7/15 past the kitchen line (away from baseline)
        if "net_left" not in r:
            r["net_left"] = kn_l + NET_FROM_KITCHEN_FRAC * (kn_l - near_l)
        if "net_right" not in r:
            r["net_right"] = kn_r + NET_FROM_KITCHEN_FRAC * (kn_r - near_r)

        net_l = r["net_left"]
        net_r = r["net_right"]

        # Far corners: reflect near corners through net if not provided
        if "far_left" not in r:
            r["far_left"] = 2.0 * net_l - near_l
        if "far_right" not in r:
            r["far_right"] = 2.0 * net_r - near_r

        far_l = r["far_left"]
        far_r = r["far_right"]

        # Far kitchen line: 7/22 from net toward far baseline
        if "kitchen_far_left" not in r:
            r["kitchen_far_left"] = net_l + KITCHEN_FRAC * (far_l - net_l)
        if "kitchen_far_right" not in r:
            r["kitchen_far_right"] = net_r + KITCHEN_FRAC * (far_r - net_r)

    # ── line / polygon construction ───────────────────────────────────────────

    def _build_geometry(self) -> None:
        r = self._raw
        self.near_left  = r["near_left"]
        self.near_right = r["near_right"]
        self.far_left   = r["far_left"]
        self.far_right  = r["far_right"]
        self.net_left   = r["net_left"]
        self.net_right  = r["net_right"]
        self._kn_l = r["kitchen_near_left"]
        self._kn_r = r["kitchen_near_right"]
        self._kf_l = r["kitchen_far_left"]
        self._kf_r = r["kitchen_far_right"]

        self.outer_polygon = np.array(
            [self.near_left, self.near_right, self.far_right, self.far_left],
            dtype=np.float32,
        )

        self.net_line      = LineModel(tuple(self.net_left),   tuple(self.net_right))
        self.left_sideline  = LineModel(tuple(self.near_left),  tuple(self.far_left))
        self.right_sideline = LineModel(tuple(self.near_right), tuple(self.far_right))
        self.near_baseline  = LineModel(tuple(self.near_left),  tuple(self.near_right))
        self.far_baseline   = LineModel(tuple(self.far_left),   tuple(self.far_right))

        self.near_kitchen_line = LineModel(tuple(self._kn_l), tuple(self._kn_r))
        self.far_kitchen_line  = LineModel(tuple(self._kf_l), tuple(self._kf_r))

        self.near_legal_polygon = np.array(
            [self._kn_l, self._kn_r, self.near_right, self.near_left],
            dtype=np.float32,
        )
        self.far_legal_polygon = np.array(
            [self._kf_l, self._kf_r, self.far_right, self.far_left],
            dtype=np.float32,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def legal_near_sign(self, ref_pt: Optional[tuple] = None) -> int:
        """
        Return +1 or -1 indicating which side of the near kitchen line is legal.
        Uses legal_ref_near anchor by default.
        """
        if ref_pt is None:
            ref_pt = tuple(self._raw["legal_ref_near"].tolist())
        d = self.near_kitchen_line.signed_distance(ref_pt)
        return 1 if d >= 0 else -1

    def anchor_dict(self) -> dict:
        return {k: v.tolist() for k, v in self._raw.items()}

    def kitchen_endpoints(self) -> dict:
        return {
            "near": (tuple(self._kn_l.tolist()), tuple(self._kn_r.tolist())),
            "far":  (tuple(self._kf_l.tolist()), tuple(self._kf_r.tolist())),
        }

    def warp(self, H: np.ndarray) -> "CourtGeometryModel":
        """Return a new model with every anchor point warped through homography H."""
        new_anchors: dict = {}
        for key, pt in self._raw.items():
            p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(p, H)
            new_anchors[key] = [float(warped[0, 0, 0]), float(warped[0, 0, 1])]
        return CourtGeometryModel(new_anchors)
