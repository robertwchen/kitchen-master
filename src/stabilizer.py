"""
Frame-to-frame stabilizer using ORB feature matching and RANSAC homography.

For each incoming frame, estimates the homography H that maps the reference
frame to the current frame. Reference kitchen line geometry is then warped
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
    ORB + BFMatcher + RANSAC homography between a reference frame and each
    incoming frame. Includes a sanity gate to reject wild transforms.
    """

    def __init__(
        self,
        n_features: int = 3000,
        ratio_test: float = 0.75,
        min_matches: int = 15,
        ransac_threshold_px: float = 4.0,
        top_mask_frac: float = 0.25,
        transform_type: str = "homography",
    ):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio_test = ratio_test
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold_px
        self.top_mask_frac = top_mask_frac
        self.transform_type = transform_type

        self._ref_kp = None
        self._ref_des = None

    def _feature_mask(self, H: int, W: int) -> np.ndarray:
        mask = np.ones((H, W), dtype=np.uint8) * 255
        mask[: int(H * self.top_mask_frac), :] = 0
        return mask

    def set_reference(self, frame: np.ndarray) -> None:
        """Detect ORB features in the reference frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        kp, des = self.orb.detectAndCompute(gray, self._feature_mask(H, W))
        self._ref_kp = kp
        self._ref_des = des
        logger.info(f"Reference frame set: {len(kp)} ORB keypoints")

    def estimate_transform(
        self, frame: np.ndarray
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
            info["status"] = "homography_failed"
            return None, info

        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        info["n_inliers"] = n_inliers

        if not self._sanity_check(H_mat):
            info["status"] = "sanity_failed"
            return None, info

        return H_mat, info

    def _sanity_check(
        self,
        H: np.ndarray,
        max_trans_px: float = 80.0,
        max_det_dev: float = 0.25,
    ) -> bool:
        tx, ty = abs(H[0, 2]), abs(H[1, 2])
        if tx > max_trans_px or ty > max_trans_px:
            logger.debug(f"Homography rejected: translation ({tx:.1f}, {ty:.1f}) too large")
            return False
        det = abs(np.linalg.det(H[:2, :2]))
        if abs(det - 1.0) > max_det_dev:
            logger.debug(f"Homography rejected: det={det:.3f}")
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
