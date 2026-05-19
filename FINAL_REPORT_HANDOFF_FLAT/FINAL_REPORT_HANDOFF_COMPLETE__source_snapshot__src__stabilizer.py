"""
Frame-to-frame stabilizer using ORB feature matching and RANSAC transforms.

For each incoming frame, estimates an affine transform or homography H that
maps the reference frame to the current frame. Reference kitchen line geometry is then warped
through H to give per-frame line positions without running Hough per frame.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameStabilizer:
    """
    ORB + BFMatcher + RANSAC transform estimation between a reference frame and
    each incoming frame. Includes a sanity gate to reject wild transforms.
    """

    def __init__(
        self,
        n_features: int = 3000,
        ratio_test: float = 0.75,
        min_matches: int = 15,
        ransac_threshold_px: float = 4.0,
        top_mask_frac: float = 0.25,
        bottom_mask_frac: float = 0.0,
        transform_type: str = "homography",
        max_translation_px: float = 80.0,
        max_det_dev: float = 0.25,
        max_rotation_deg: float | None = None,
        max_scale_dev: float | None = None,
    ):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio_test = ratio_test
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold_px
        self.top_mask_frac = top_mask_frac
        self.bottom_mask_frac = bottom_mask_frac
        self.transform_type = transform_type
        self.max_translation_px = max_translation_px
        self.max_det_dev = max_det_dev
        self.max_rotation_deg = max_rotation_deg
        self.max_scale_dev = max_scale_dev

        self._ref_kp = None
        self._ref_des = None
        self._custom_mask: Optional[np.ndarray] = None

    def _feature_mask(self, H: int, W: int) -> np.ndarray:
        mask = np.ones((H, W), dtype=np.uint8) * 255
        mask[: int(H * self.top_mask_frac), :] = 0
        if self.bottom_mask_frac > 0.0:
            mask[int(H * (1.0 - self.bottom_mask_frac)) :, :] = 0
        if self._custom_mask is not None:
            custom = self._custom_mask
            if custom.shape[:2] != (H, W):
                custom = cv2.resize(custom, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_and(mask, custom)
        return mask

    def set_feature_mask(self, mask: Optional[np.ndarray]) -> None:
        """Set an optional binary ROI mask for feature detection/matching."""
        if mask is None:
            self._custom_mask = None
            return
        if mask.ndim != 2:
            raise ValueError("Feature mask must be a single-channel uint8 image")
        self._custom_mask = (mask > 0).astype(np.uint8) * 255

    def set_reference(self, frame: np.ndarray) -> None:
        """Detect ORB features in the reference frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        kp, des = self.orb.detectAndCompute(gray, self._feature_mask(H, W))
        self._ref_kp = kp
        self._ref_des = des
        logger.info(f"Reference frame set: {len(kp)} ORB keypoints")

    def estimate_transform(
        self, frame: np.ndarray, update_ref_on_success: bool = False
    ) -> tuple[Optional[np.ndarray], dict]:
        """
        Estimate 3×3 homography from reference → current frame.

        Returns (H, info) where H is None if estimation fails or fails sanity.
        info contains diagnostic counts and a status string.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H_frame, W_frame = gray.shape
        kp, des = self.orb.detectAndCompute(gray, self._feature_mask(H_frame, W_frame))

        info: dict = {
            "n_keypoints": len(kp),
            "n_matches": 0,
            "n_inliers": 0,
            "inlier_ratio": 0.0,
            "status": "ok",
        }

        if des is None or self._ref_des is None:
            info["status"] = "no_descriptors"
            return None, info

        if len(kp) < self.min_matches:
            info["status"] = "insufficient_keypoints"
            return None, info

        raw = self.matcher.knnMatch(self._ref_des, des, k=2)
        good = [
            m
            for pair in raw
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < self.ratio_test * n.distance
        ]
        info["n_matches"] = len(good)

        if len(good) < self.min_matches:
            info["status"] = "insufficient_matches"
            return None, info

        src_pts = np.float32(
            [self._ref_kp[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        if self.transform_type == "affine":
            mat, inlier_mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
            )
            if mat is not None:
                H_mat = np.eye(3, dtype=np.float64)
                H_mat[:2, :] = mat
            else:
                H_mat = None
        else:
            H_mat, inlier_mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold
            )

        if H_mat is None:
            info["status"] = f"{self.transform_type}_failed"
            return None, info

        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        info["n_inliers"] = n_inliers
        info["inlier_ratio"] = round(n_inliers / max(1, len(good)), 4)

        if not self._sanity_check(H_mat):
            info["status"] = "sanity_failed"
            return None, info

        if update_ref_on_success:
            self._ref_kp = kp
            self._ref_des = des

        return H_mat, info

    def _sanity_check(
        self,
        H: np.ndarray,
    ) -> bool:
        tx, ty = abs(H[0, 2]), abs(H[1, 2])
        if tx > self.max_translation_px or ty > self.max_translation_px:
            logger.debug(f"Homography rejected: translation ({tx:.1f}, {ty:.1f}) too large")
            return False
        det = abs(np.linalg.det(H[:2, :2]))
        if abs(det - 1.0) > self.max_det_dev:
            logger.debug(f"Homography rejected: det={det:.3f}")
            return False
        scale = float(np.sqrt(H[0, 0] * H[0, 0] + H[0, 1] * H[0, 1]))
        if self.max_scale_dev is not None and abs(scale - 1.0) > self.max_scale_dev:
            logger.debug(f"Transform rejected: scale={scale:.4f}")
            return False
        if self.max_rotation_deg is not None:
            rotation_deg = abs(float(np.degrees(np.arctan2(-H[0, 1], H[0, 0]))))
            if rotation_deg > self.max_rotation_deg:
                logger.debug(f"Transform rejected: rotation={rotation_deg:.3f}deg")
                return False
        return True

    @staticmethod
    def warp_point(pt: tuple, H: np.ndarray) -> tuple[float, float]:
        """Map a single (x, y) point through a homography."""
        p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(p, H)
        return float(warped[0, 0, 0]), float(warped[0, 0, 1])

    @staticmethod
    def warp_line(
        p1: tuple, p2: tuple, H: np.ndarray
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Map a line segment (two endpoints) through a homography."""
        pts = np.array(
            [[[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]]],
            dtype=np.float32,
        )
        warped = cv2.perspectiveTransform(pts, H)
        wp1 = (float(warped[0, 0, 0]), float(warped[0, 0, 1]))
        wp2 = (float(warped[0, 1, 0]), float(warped[0, 1, 1]))
        return wp1, wp2


def refine_line_roi(
    frame: np.ndarray,
    p1: tuple,
    p2: tuple,
    search_px: int = 20,
    n_sample_points: int = 30,
) -> int:
    """
    Search ±search_px perpendicular to the line (p1→p2) for the offset that
    maximizes horizontal Sobel edge response. Returns the best integer offset.

    The perpendicular direction is the line normal (a, b) from ax+by+c=0.
    """
    H_frame, W_frame = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

    # Normal vector of the line (a, b) from normalized form
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    norm = np.sqrt(dx * dx + dy * dy)
    if norm < 1e-9:
        return 0
    na, nb = dy / norm, -dx / norm  # unit normal

    t_vals = np.linspace(0.0, 1.0, n_sample_points)
    base_x = p1[0] + t_vals * (p2[0] - p1[0])
    base_y = p1[1] + t_vals * (p2[1] - p1[1])

    best_offset, best_score = 0, -1.0
    for offset in range(-search_px, search_px + 1):
        sx = base_x + offset * na
        sy = base_y + offset * nb
        xi = np.clip(np.round(sx).astype(int), 0, W_frame - 1)
        yi = np.clip(np.round(sy).astype(int), 0, H_frame - 1)
        score = float(sobel[yi, xi].mean())
        if score > best_score:
            best_score = score
            best_offset = offset

    return best_offset
