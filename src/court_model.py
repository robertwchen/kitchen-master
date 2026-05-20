"""
Pickleball court geometry model for a side-facing camera showing the kitchen zone.

The camera faces the court from the side (perpendicular to the baselines).
From this view the kitchen appears as a rectangle:

        far-L ──────────────── far-R      ← far kitchen line (back edge)
          |                      |
  LEFT    |     KITCHEN          |   RIGHT
  LEGAL   |    (illegal zone)    |   LEGAL
  ZONE    |                      |   ZONE
          |                      |
       near-L ──────────────── near-R     ← near kitchen line (front edge)

The actual NVZ boundary lines visible from this camera are:
  LEFT  boundary: near-L → far-L  (left edge of kitchen rectangle)
  RIGHT boundary: near-R → far-R  (right edge of kitchen rectangle)

Legal zones (green fill) extend outward:
  left_legal_polygon  — to the LEFT  of the left  boundary line
  right_legal_polygon — to the RIGHT of the right boundary line

near_kitchen_line and far_kitchen_line are also stored for display and
ORB stabilization (they are the most prominent horizontal lines in the image).
"""

import cv2
import numpy as np
from typing import Optional

from src.court_registration import LineModel


class CourtGeometryModel:
    """
    Court geometry from annotated kitchen-rectangle corners.

    Required anchors (2)
    --------------------
    kitchen_near_left    Left  end of the near (front) kitchen line.
    kitchen_near_right   Right end of the near (front) kitchen line.

    Optional anchors (2)
    --------------------
    kitchen_far_left     Left  end of the far (back) kitchen line.
    kitchen_far_right    Right end of the far (back) kitchen line.

    When all four corners are present, the model builds the left and right
    NVZ boundary lines and their outward legal-zone polygons.
    Any legacy 'legal_ref_near' key in the anchor dict is silently preserved
    (for round-trip warp compatibility) but not used in geometry.
    """

    REQUIRED = {
        "kitchen_near_left",
        "kitchen_near_right",
    }

    def __init__(self, anchors: dict) -> None:
        missing = self.REQUIRED - set(anchors.keys())
        if missing:
            raise ValueError(f"Missing required anchors: {sorted(missing)}")

        self._raw: dict[str, np.ndarray] = {
            k: np.array(v, dtype=float) for k, v in anchors.items()
        }
        self._build_geometry()

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_geometry(self) -> None:
        r = self._raw
        self._kn_l = r["kitchen_near_left"]
        self._kn_r = r["kitchen_near_right"]

        # Near and far kitchen lines (horizontal edges — for display and ORB)
        self.near_kitchen_line = LineModel(tuple(self._kn_l), tuple(self._kn_r))

        has_far = "kitchen_far_left" in r and "kitchen_far_right" in r
        if has_far:
            self._kf_l = r["kitchen_far_left"]
            self._kf_r = r["kitchen_far_right"]
            self.far_kitchen_line: Optional[LineModel] = LineModel(
                tuple(self._kf_l), tuple(self._kf_r)
            )

            # NVZ boundary lines (the lines players must not cross to volley)
            # LEFT:  near-left corner → far-left corner
            # RIGHT: near-right corner → far-right corner
            self.left_boundary_line: Optional[LineModel] = LineModel(
                tuple(self._kn_l), tuple(self._kf_l)
            )
            # Orient both side boundaries so signed_distance > 0 means the
            # legal/outside side. For the right boundary that means far -> near.
            self.right_boundary_line: Optional[LineModel] = LineModel(
                tuple(self._kf_r), tuple(self._kn_r)
            )

            # Kitchen centre — used to determine which side of each boundary
            # is inside the kitchen (illegal) so we can shade the outside (legal).
            kitchen_center = (
                self._kn_l + self._kn_r + self._kf_l + self._kf_r
            ) / 4.0

            self.left_legal_polygon: Optional[np.ndarray] = self._side_polygon(
                self._kn_l, self._kf_l, self.left_boundary_line, kitchen_center
            )
            self.right_legal_polygon: Optional[np.ndarray] = self._side_polygon(
                self._kn_r, self._kf_r, self.right_boundary_line, kitchen_center
            )
        else:
            self._kf_l = self._kf_r = None
            self.far_kitchen_line = None
            self.left_boundary_line = None
            self.right_boundary_line = None
            self.left_legal_polygon = None
            self.right_legal_polygon = None

    @staticmethod
    def _side_polygon(
        pt_near: np.ndarray,
        pt_far: np.ndarray,
        boundary: LineModel,
        kitchen_center: np.ndarray,
        lateral: float = 5000.0,
        perp: float = 5000.0,
    ) -> np.ndarray:
        """
        Build a large polygon that covers the legal zone on one side of a
        boundary line.  Extends LATERAL px along the line and PERP px
        outward (away from the kitchen centre).
        """
        # Unit vector along the boundary (near → far)
        line_dir = pt_far - pt_near
        length = np.linalg.norm(line_dir)
        if length > 1e-9:
            line_dir = line_dir / length

        # Stretch endpoints well past the frame in both directions
        ext_near = pt_near - line_dir * lateral
        ext_far  = pt_far  + line_dir * lateral

        # Normal pointing AWAY from the kitchen (outward = legal side)
        inside_d = boundary.signed_distance(tuple(kitchen_center))
        outside_sign = -1.0 if inside_d >= 0 else 1.0
        na = boundary.a * outside_sign
        nb = boundary.b * outside_sign
        nn = np.sqrt(na * na + nb * nb)
        if nn > 1e-9:
            na /= nn
            nb /= nn
        out_vec = np.array([na * perp, nb * perp])

        return np.array(
            [ext_near, ext_far, ext_far + out_vec, ext_near + out_vec],
            dtype=np.float32,
        )

    # ── public API ────────────────────────────────────────────────────────────

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
