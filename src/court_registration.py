"""
Court registration for fixed-camera NVZ line detection.

Loads manually-annotated reference line endpoints, fits line equations,
and propagates fixed geometry through the whole clip. For a static camera,
the same line parameters apply to every frame.

Stability check: measures Sobel edge response in a band along each registered
line across randomly sampled frames to verify the geometry holds.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Line model
# ---------------------------------------------------------------------------

class LineModel:
    """
    Line defined by two points with normalized form ax + by + c = 0.

    Sign convention for signed_distance:
        positive → same side as the direction of the normal (a, b)
        negative → opposite side
    """

    def __init__(self, p1: tuple, p2: tuple):
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)

        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]

        # Line through two points: dy·x − dx·y + (dx·y1 − dy·x1) = 0
        self.a = float(dy)
        self.b = float(-dx)
        self.c = float(dx * self.p1[1] - dy * self.p1[0])

        norm = np.sqrt(self.a ** 2 + self.b ** 2)
        if norm > 1e-9:
            self.a /= norm
            self.b /= norm
            self.c /= norm

    def signed_distance(self, point: tuple) -> float:
        """Signed perpendicular distance from point to line (pixels)."""
        return self.a * point[0] + self.b * point[1] + self.c

    def y_at_x(self, x: float) -> Optional[float]:
        if abs(self.b) < 1e-9:
            return None
        return -(self.a * x + self.c) / self.b

    def x_at_y(self, y: float) -> Optional[float]:
        if abs(self.a) < 1e-9:
            return None
        return -(self.b * y + self.c) / self.a

    def endpoints_in_frame(self, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return two pixel endpoints that span the full frame width."""
        candidates = []

        for x in (0, width - 1):
            y = self.y_at_x(float(x))
            if y is not None and 0 <= y < height:
                candidates.append((int(x), int(round(y))))

        for y in (0, height - 1):
            x = self.x_at_y(float(y))
            if x is not None and 0 <= x < width:
                candidates.append((int(round(x)), int(y)))

        unique = list({(p[0], p[1]) for p in candidates})
        if len(unique) < 2:
            return (int(self.p1[0]), int(self.p1[1])), (int(self.p2[0]), int(self.p2[1]))

        unique.sort(key=lambda p: p[0])
        return unique[0], unique[-1]

    def to_dict(self) -> dict:
        return {
            "p1": self.p1.tolist(),
            "p2": self.p2.tolist(),
            "a": round(self.a, 8),
            "b": round(self.b, 8),
            "c": round(self.c, 4),
        }


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def _fit_line_from_frames(frames_data: list[dict], key: str) -> LineModel:
    """Average endpoints across annotated frames and return a LineModel."""
    p1s, p2s = [], []
    for f in frames_data:
        if key in f:
            p1s.append(f[key]["p1"])
            p2s.append(f[key]["p2"])
    if not p1s:
        raise ValueError(f"No annotation found for '{key}'")
    p1 = np.mean(p1s, axis=0)
    p2 = np.mean(p2s, axis=0)
    return LineModel(tuple(p1), tuple(p2))


# ---------------------------------------------------------------------------
# Court registration
# ---------------------------------------------------------------------------

class CourtRegistration:
    """
    Registers NVZ kitchen line geometry from a reference annotation file.

    For a static camera, fit() is called once and the same geometry is
    applied to every frame.
    """

    def __init__(self, annotation_path: Path):
        with open(annotation_path) as f:
            self.annotations = json.load(f)

        self.near_line: Optional[LineModel] = None
        self.far_line: Optional[LineModel] = None
        self.legal_ref_point: Optional[tuple] = None
        self._fitted = False

    def fit(self) -> None:
        """Fit line models from all annotated reference frames."""
        frames_data = self.annotations["annotated_frames"]
        all_keys = {k for f in frames_data for k in f}

        if "near_kitchen_line" in all_keys:
            self.near_line = _fit_line_from_frames(frames_data, "near_kitchen_line")
            logger.info(f"Near line: p1={self.near_line.p1.tolist()}  p2={self.near_line.p2.tolist()}")

        if "far_kitchen_line" in all_keys:
            self.far_line = _fit_line_from_frames(frames_data, "far_kitchen_line")
            logger.info(f"Far  line: p1={self.far_line.p1.tolist()}  p2={self.far_line.p2.tolist()}")

        # Legal-side reference point
        for f in frames_data:
            if "legal_side_reference_point" in f:
                self.legal_ref_point = tuple(f["legal_side_reference_point"])
                break

        if self.near_line is None and self.far_line is None:
            raise ValueError("Annotation has no kitchen line data. Check your JSON format.")

        self._fitted = True
        logger.info("Court registration fitted.")

    def legal_side_sign(self) -> Optional[int]:
        """
        Return +1 or -1 indicating which sign of signed_distance is 'legal'
        for the near kitchen line, based on the reference point.
        """
        if self.near_line is None or self.legal_ref_point is None:
            return None
        d = self.near_line.signed_distance(self.legal_ref_point)
        return 1 if d >= 0 else -1

    def refine(self, cap: cv2.VideoCapture, n_frames: int = 5, search_px: int = 10) -> dict:
        """
        Shift each line by ±search_px to maximise edge response.

        Reads n_frames evenly spaced frames from cap. Returns dict with
        refinement offsets (in pixels, perpendicular to each line).
        cap is left open after this call.
        """
        assert self._fitted
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        results = {}

        for line_name, line in [("near", self.near_line), ("far", self.far_line)]:
            if line is None:
                continue

            best_offset, best_score = 0, -1.0
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            for offset in range(-search_px, search_px + 1):
                shifted = LineModel(
                    (line.p1[0] + offset * line.a, line.p1[1] + offset * line.b),
                    (line.p2[0] + offset * line.a, line.p2[1] + offset * line.b),
                )
                pt1, pt2 = shifted.endpoints_in_frame(W, H)
                scores = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.line(mask, pt1, pt2, 255, 5)
                    scores.append(float(sobel[mask > 0].mean()))
                if scores:
                    score = float(np.mean(scores))
                    if score > best_score:
                        best_score = score
                        best_offset = offset

            # Apply refinement
            if line_name == "near" and self.near_line is not None:
                self.near_line = LineModel(
                    (self.near_line.p1[0] + best_offset * self.near_line.a,
                     self.near_line.p1[1] + best_offset * self.near_line.b),
                    (self.near_line.p2[0] + best_offset * self.near_line.a,
                     self.near_line.p2[1] + best_offset * self.near_line.b),
                )
            elif line_name == "far" and self.far_line is not None:
                self.far_line = LineModel(
                    (self.far_line.p1[0] + best_offset * self.far_line.a,
                     self.far_line.p1[1] + best_offset * self.far_line.b),
                    (self.far_line.p2[0] + best_offset * self.far_line.a,
                     self.far_line.p2[1] + best_offset * self.far_line.b),
                )

            results[line_name] = {"refinement_offset_px": best_offset, "peak_edge_score": round(best_score, 2)}
            logger.info(f"  {line_name} line refined by {best_offset:+d}px  (edge score={best_score:.2f})")

        return results

    def stability_check(self, cap: cv2.VideoCapture, n_samples: int = 20) -> dict:
        """
        Sample n_samples frames and measure edge response along each registered line.

        Returns per-line statistics: mean, std, and coefficient of variation.
        Low CV → geometry is stable across the clip.
        """
        assert self._fitted
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        indices = np.linspace(0, total - 1, min(n_samples, total), dtype=int)

        results = {}
        for line_name, line in [("near", self.near_line), ("far", self.far_line)]:
            if line is None:
                continue
            pt1, pt2 = line.endpoints_in_frame(W, H)
            strengths = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.line(mask, pt1, pt2, 255, 7)
                strengths.append(float(sobel[mask > 0].mean()))

            if strengths:
                mean_s = float(np.mean(strengths))
                std_s = float(np.std(strengths))
                results[line_name] = {
                    "mean_edge_strength": round(mean_s, 2),
                    "std_edge_strength": round(std_s, 2),
                    "cv": round(std_s / (mean_s + 1e-6), 4),
                    "n_frames_sampled": len(strengths),
                    "assessment": "stable" if std_s / (mean_s + 1e-6) < 0.15 else "check",
                }
                logger.info(
                    f"  {line_name} stability: mean={mean_s:.1f}  std={std_s:.1f}  "
                    f"cv={results[line_name]['cv']:.3f}  → {results[line_name]['assessment']}"
                )

        return results

    def csv_row(self, frame_index: int, timestamp_s: float) -> dict:
        """Return one CSV row of line parameters for a given frame."""
        row: dict = {"frame_index": frame_index, "timestamp_s": round(timestamp_s, 4)}
        for name, line in [("near", self.near_line), ("far", self.far_line)]:
            if line is not None:
                row[f"{name}_p1_x"] = round(line.p1[0], 1)
                row[f"{name}_p1_y"] = round(line.p1[1], 1)
                row[f"{name}_p2_x"] = round(line.p2[0], 1)
                row[f"{name}_p2_y"] = round(line.p2[1], 1)
                row[f"{name}_a"] = round(line.a, 8)
                row[f"{name}_b"] = round(line.b, 8)
                row[f"{name}_c"] = round(line.c, 4)
        return row
