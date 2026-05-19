"""
Lightweight learned ball detector helpers.

This module adds an optional detector-first Stage 2 backend built around
Ultralytics models. It is intentionally isolated from the rest of the pipeline:

- import Ultralytics lazily so the classical CV pipeline still works without it
- keep outputs in the same candidate format used by the blob tracker
- preserve raw detection points so Stage 3 can anchor bounces to real detections
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", " ").replace("_", " ")


@dataclass
class UltralyticsBallDetector:
    model: Any
    conf: float
    iou: float
    imgsz: int
    max_det: int
    device: Optional[str]
    class_ids: Optional[list[int]]
    class_names: Optional[set[str]]
    use_tiled_inference: bool
    tile_size: int
    tile_overlap: float
    merge_radius_px: float

    @classmethod
    def from_config(cls, cfg: dict) -> "UltralyticsBallDetector":
        ultra_cfg = cfg.get("ultralytics", {}) if isinstance(cfg.get("ultralytics"), dict) else {}
        model_path = ultra_cfg.get("model_path") or cfg.get("ultralytics_model_path")
        if not model_path:
            raise ValueError(
                "Ultralytics backend requires ball_tracking.ultralytics.model_path"
            )

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics backend requested, but 'ultralytics' is not installed. "
                "Install it with: pip install ultralytics"
            ) from exc

        model = YOLO(str(Path(model_path)))
        class_ids = ultra_cfg.get("class_ids")
        class_names_cfg = ultra_cfg.get("class_names")
        class_names = (
            {_normalize_name(str(name)) for name in class_names_cfg}
            if class_names_cfg else None
        )

        return cls(
            model=model,
            conf=float(ultra_cfg.get("confidence", 0.15)),
            iou=float(ultra_cfg.get("iou", 0.45)),
            imgsz=int(ultra_cfg.get("imgsz", 1280)),
            max_det=int(ultra_cfg.get("max_det", 8)),
            device=str(ultra_cfg.get("device")) if ultra_cfg.get("device") is not None else None,
            class_ids=[int(x) for x in class_ids] if class_ids else None,
            class_names=class_names,
            use_tiled_inference=bool(ultra_cfg.get("use_tiled_inference", True)),
            tile_size=int(ultra_cfg.get("tile_size", 960)),
            tile_overlap=float(ultra_cfg.get("tile_overlap", 0.25)),
            merge_radius_px=float(ultra_cfg.get("merge_radius_px", 24.0)),
        )

    def _predict(self, image: np.ndarray) -> Any:
        kwargs: dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        if self.class_ids is not None:
            kwargs["classes"] = self.class_ids
        return self.model.predict(image, **kwargs)[0]

    def _parse_result(self, result: Any, offset_x: int, offset_y: int) -> list[dict]:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        candidates: list[dict] = []
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            x0, y0, x1, y1 = [float(v) for v in xyxy]
            x0 += offset_x
            x1 += offset_x
            y0 += offset_y
            y1 += offset_y

            cls_id = None
            if getattr(box, "cls", None) is not None:
                cls_id = int(float(box.cls[0]))
            class_name = str(names.get(cls_id, cls_id if cls_id is not None else "unknown"))
            if self.class_names is not None and _normalize_name(class_name) not in self.class_names:
                continue

            conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            w = max(1.0, x1 - x0)
            h = max(1.0, y1 - y0)
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            radius = max(2.0, min(w, h) / 2.0)

            candidates.append({
                "x": cx,
                "y": cy,
                "radius": radius,
                "area": w * h,
                "circularity": None,
                "v_at_center": None,
                "score": conf,
                "detector_confidence": conf,
                "bbox": [x0, y0, x1, y1],
                "class_name": class_name,
                "source": "ultralytics",
            })
        return candidates

    def _merge_candidates(self, candidates: list[dict]) -> list[dict]:
        merged: list[dict] = []
        for cand in sorted(candidates, key=lambda c: c["score"], reverse=True):
            duplicate = False
            for kept in merged:
                d = float(np.hypot(cand["x"] - kept["x"], cand["y"] - kept["y"]))
                if d <= self.merge_radius_px:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(cand)
        return merged

    def detect(self, frame: np.ndarray) -> list[dict]:
        H, W = frame.shape[:2]
        if not self.use_tiled_inference or self.tile_size >= max(H, W):
            result = self._predict(frame)
            return self._merge_candidates(self._parse_result(result, 0, 0))

        tile_step = max(1, int(self.tile_size * (1.0 - self.tile_overlap)))
        candidates: list[dict] = []
        y_positions = list(range(0, max(1, H - self.tile_size + 1), tile_step))
        x_positions = list(range(0, max(1, W - self.tile_size + 1), tile_step))
        if not y_positions or y_positions[-1] != max(0, H - self.tile_size):
            y_positions.append(max(0, H - self.tile_size))
        if not x_positions or x_positions[-1] != max(0, W - self.tile_size):
            x_positions.append(max(0, W - self.tile_size))

        for y0 in y_positions:
            for x0 in x_positions:
                tile = frame[y0:min(H, y0 + self.tile_size), x0:min(W, x0 + self.tile_size)]
                result = self._predict(tile)
                candidates.extend(self._parse_result(result, x0, y0))
        return self._merge_candidates(candidates)


def render_ultralytics_debug(frame: np.ndarray, candidates: list[dict]) -> np.ndarray:
    """Render a detector-side debug panel that mirrors the old mask view."""
    panel = np.zeros_like(frame)
    for i, cand in enumerate(candidates[:12], start=1):
        x0, y0, x1, y1 = [int(round(v)) for v in cand.get("bbox", [cand["x"], cand["y"], cand["x"], cand["y"]])]
        cv2.rectangle(panel, (x0, y0), (x1, y1), (0, 255, 255), 2)
        label = f"{i}:{cand.get('class_name', 'ball')} {cand.get('detector_confidence', cand.get('score', 0.0)):.2f}"
        cv2.putText(panel, label, (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.circle(panel, (int(round(cand["x"])), int(round(cand["y"]))), 3, (255, 255, 255), -1)
    return panel
