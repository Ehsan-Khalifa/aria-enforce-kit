"""
ARIA-Enforce — Rider-attribute model wrapper (helmet / seatbelt / triple-riding).

Wraps the YOLO model trained in Phase 1. Crucially, it is STUB-CAPABLE:
if the trained weights are not present yet (training still running, or a
reviewer who didn't download them), it returns an empty result and reports
status="model_pending" instead of crashing. This lets every other phase be
built and demoed today, and the real model drops in with zero code changes.

Output contract (per uploaded image):
    RiderInference(
        status="ok" | "model_pending" | "error",
        riders=[RiderBox(bbox, has_helmet, rider_count, has_seatbelt, conf), ...],
        raw_boxes=[...]   # all detections for annotation/debug
    )
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import RIDER_MODEL_CLASSES, RIDER_MODEL_WEIGHTS

logger = logging.getLogger("aria.enforce.rider")


@dataclass
class RiderBox:
    bbox:         list[int]            # [x1,y1,x2,y2]
    kind:         str                  # class name from RIDER_MODEL_CLASSES
    confidence:   float


@dataclass
class RiderInference:
    status:    str                     # ok | model_pending | error
    boxes:     list[RiderBox] = field(default_factory=list)
    message:   str = ""


class RiderAttributeModel:
    def __init__(self, weights: str = RIDER_MODEL_WEIGHTS,
                 conf_threshold: float = 0.35, device: str = "auto"):
        self.weights = Path(weights)
        self.conf = conf_threshold
        self.device = device
        self._model = None
        self._status = "uninit"
        self._load()

    def _load(self):
        if not self.weights.exists():
            self._status = "model_pending"
            logger.warning(
                f"[RiderModel] weights not found at {self.weights} — "
                f"running in STUB mode (helmet/seatbelt checks skipped). "
                f"Drop trained best.pt here to activate.")
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights))
            self._status = "ok"
            logger.info(f"[RiderModel] loaded {self.weights}")
        except Exception as e:               # pragma: no cover
            self._status = "error"
            logger.error(f"[RiderModel] failed to load: {e}")

    @property
    def ready(self) -> bool:
        return self._status == "ok"

    def infer(self, frame: np.ndarray) -> RiderInference:
        if self._status != "ok":
            return RiderInference(status=self._status,
                                  message="rider-attribute model not loaded")
        try:
            res = self._model.predict(frame, conf=self.conf, verbose=False)[0]
            names = res.names
            boxes: list[RiderBox] = []
            for b in res.boxes:
                cls_id = int(b.cls[0])
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                boxes.append(RiderBox(bbox=xyxy,
                                      kind=names.get(cls_id, str(cls_id)),
                                      confidence=float(b.conf[0])))
            return RiderInference(status="ok", boxes=boxes)
        except Exception as e:               # pragma: no cover
            logger.error(f"[RiderModel] inference error: {e}")
            return RiderInference(status="error", message=str(e))
