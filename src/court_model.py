"""
Pickleball court geometry model derived from anchor points.

Anchor points define the court structure in reference-frame pixel coordinates.
All derived geometry (kitchen lines, legal zones, court polygon) is computed
from these anchors. CourtGeometryModel.warp(H) propagates the entire model
through a homography so the same structure works for any registered frame.
"""

import cv2
import numpy as np
from typing import Optional

from src.court_registration import LineModel

# Kitchen line is 7 ft from net; each half-court is 22 ft deep.
KITCHEN_FRACTION: float = 7.0 / 22.0


class CourtGeometryModel:
    """
    Pickleball court geometry anchored in pixel coordinates.

    Required anchors
    ----------------
    near_left    Bottom-left corner of the pickleball court (near-camera side).
    near_right   Bottom-right corner (near-camera side).
    far_left     Top-left corner (far side, behind net).
    far_right    Top-right corner (far side).
    net_left     Left anchor of the net (where net meets left sideline).
    net_right    Right anchor of the net.

    Optional anchors (override proportional inference)
    --------------------------------------------------
    kitchen_near_left    Near kitchen line ∩ left sideline.
    kitchen_near_right   Near kitchen line ∩ right sideline.
    kitchen_far_left     Far kitchen line ∩ left sideline.
    kitchen_far_right    Far kitchen line ∩ right sideline.
    legal_ref_near       A point clearly behind the near kitchen line (legal side).
    """

    REQUIRED = {
        "near_left", "near_right", "far_left", "far_right", "net_left", "net_right"
    }

    def __init__(self, anchors: dict) -> None:
        missing = self.REQUIRED - set(anchors.keys())
        if missing:
            raise ValueError(f"Missing required anchors: {sorted(missing)}")

        # Store anchors as float numpy arrays
        self._raw: dict[str, np.ndarray] = {
            k: np.array(v, dtype=float) for k, v in anchors.items()
        }
        self._recompute()

    # ── internal ─────────────────────────────────────────────────────────────

    def _recompute(self) -> None:
        r = self._raw
        self.near_left = r["near_left"]
        self.near_right = r["near_right"]
        self.far_left = r["far_left"]
        self.far_right = r["far_right"]
        self.net_left = r["net_left"]
        self.net_right = r["net_right"]

        # Court polygon (near_left → near_right → far_right → far_left)
        self.outer_polygon = np.array(
            [self.near_left, self.near_right, self.far_right, self.far_left],
            dtype=np.float32,
        )

        # Structural lines
        self.net_line = LineModel(tuple(self.net_left), tuple(self.net_right))
        self.left_sideline = LineModel(tuple(self.near_left), tuple(self.far_left))
        self.right_sideline = LineModel(tuple(self.near_right), tuple(self.far_right))
        self.near_baseline = LineModel(tuple(self.near_left), tuple(self.near_right))
        self.far_baseline = LineModel(tuple(self.far_left), tuple(self.far_right))

        # Near kitchen line
        if "kitchen_near_left" in r and "kitchen_near_right" in r:
            kn_l = r["kitchen_near_left"]
            kn_r = r["kitchen_near_right"]
        else:
            kn_l = self.net_left + KITCHEN_FRACTION * (self.near_left - self.net_left)
            kn_r = self.net_right + KITCHEN_FRACTION * (self.near_right - self.net_right)
        self._raw.setdefault("kitchen_near_left", kn_l)
        self._raw.setdefault("kitchen_near_right", kn_r)
        self.near_kitchen_line = LineModel(tuple(kn_l), tuple(kn_r))
        self._kn_l, self._kn_r = kn_l, kn_r

        # Far kitchen line
        if "kitchen_far_left" in r and "kitchen_far_right" in r:
            kf_l = r["kitchen_far_left"]
            kf_r = r["kitchen_far_right"]
        else:
            kf_l = self.net_left + KITCHEN_FRACTION * (self.far_left - self.net_left)
            kf_r = self.net_right + KITCHEN_FRACTION * (self.far_right - self.net_right)
        self._raw.setdefault("kitchen_far_left", kf_l)
        self._raw.setdefault("kitchen_far_right", kf_r)
        self.far_kitchen_line = LineModel(tuple(kf_l), tuple(kf_r))
        self._kf_l, self._kf_r = kf_l, kf_r

        # Legal zone polygons (filled behind each kitchen line)
        self.near_legal_polygon = np.array(
            [kn_l, kn_r, self.near_right, self.near_left], dtype=np.float32
        )
        self.far_legal_polygon = np.array(
            [kf_l, kf_r, self.far_right, self.far_left], dtype=np.float32
        )

    # ── public API ────────────────────────────────────────────────────────────

    def legal_near_sign(self, ref_pt: tuple) -> int:
        """Return +1/-1 indicating which side of the near kitchen line is legal."""
        d = self.near_kitchen_line.signed_distance(ref_pt)
        return 1 if d >= 0 else -1

    def anchor_dict(self) -> dict:
        """Return all effective anchor positions as plain Python lists."""
        return {k: v.tolist() for k, v in self._raw.items()}

    def warp(self, H: np.ndarray) -> "CourtGeometryModel":
        """Return a new model with every anchor point warped through homography H."""
        new_anchors: dict = {}
        for key, pt in self._raw.items():
            p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(p, H)
            new_anchors[key] = [float(warped[0, 0, 0]), float(warped[0, 0, 1])]
        return CourtGeometryModel(new_anchors)

    def kitchen_endpoints(self) -> dict:
        """
        Return the two kitchen line segments as (p1, p2) tuples.
        """
        return {
            "near": (tuple(self._kn_l.tolist()), tuple(self._kn_r.tolist())),
            "far": (tuple(self._kf_l.tolist()), tuple(self._kf_r.tolist())),
        }
