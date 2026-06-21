"""
ARIA — Vehicle Detector
Wraps YOLOv8 (or ONNX for Pi) for inference.
Returns structured detections ready for the tracker.
"""

from __future__ import annotations
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

CLASS_NAMES = [
    "two_wheeler", "auto_rickshaw", "e_rickshaw", "car_suv",
    "tempo_mini_truck", "bus", "heavy_vehicle", "tractor",
    "bicycle", "pedestrian", "animal_cart",
]

@dataclass
class RawDetection:
    bbox:          list[int]   # [x1, y1, x2, y2] pixels
    confidence:    float
    class_id:      int
    vehicle_class: str


class VehicleDetector:
    """
    Unified detector: uses Ultralytics YOLO on GPU/CPU, or ONNX on Raspberry Pi.

    Usage:
        det = VehicleDetector("aria/models/detection/yolov8n_indian/weights/best.pt")
        results = det.detect(frame)   # numpy BGR frame
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.40,
                 iou_threshold: float = 0.45, device: str = "auto"):
        self.conf = conf_threshold
        self.iou  = iou_threshold
        self._model_path = Path(model_path)
        self._device = device
        self._fps_log: list[float] = []

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Train first: python aria/train.py"
            )

        ext = self._model_path.suffix.lower()
        if ext == ".onnx":
            self._backend = "onnx"
            self._load_onnx()
        else:
            self._backend = "ultralytics"
            self._load_ultralytics()

        print(f"[ARIA Detector] Loaded {self._model_path.name} "
              f"via {self._backend} backend on {self._device}")

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_ultralytics(self):
        from ultralytics import YOLO
        self._model = YOLO(str(self._model_path))
        if self._device == "auto":
            try:
                import torch
                self._device = "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"

    def _load_onnx(self):
        import onnxruntime as ort
        # Use CPU for Pi; use CUDA if available
        providers = ["CPUExecutionProvider"]
        try:
            import torch
            if torch.cuda.is_available():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except ImportError:
            pass
        self._session = ort.InferenceSession(str(self._model_path),
                                             providers=providers)
        self._input_name  = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape  # [1,3,640,640]
        self._device = "onnx-cpu"

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        """Run detection on a BGR numpy frame. Returns list of RawDetection."""
        t0 = time.perf_counter()

        if self._backend == "ultralytics":
            dets = self._detect_ultralytics(frame)
        else:
            dets = self._detect_onnx(frame)

        elapsed = time.perf_counter() - t0
        self._fps_log.append(1.0 / elapsed if elapsed > 0 else 0)
        if len(self._fps_log) > 100:
            self._fps_log.pop(0)

        return dets

    def _detect_ultralytics(self, frame: np.ndarray) -> list[RawDetection]:
        results = self._model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self._device,
            verbose=False,
        )
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf    = float(box.conf[0])
                cls_id  = int(box.cls[0])
                if cls_id >= len(CLASS_NAMES):
                    continue
                dets.append(RawDetection(
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    class_id=cls_id,
                    vehicle_class=CLASS_NAMES[cls_id],
                ))
        return dets

    def _detect_onnx(self, frame: np.ndarray) -> list[RawDetection]:
        """ONNX inference path for Raspberry Pi."""
        import cv2
        h_orig, w_orig = frame.shape[:2]
        target = self._input_shape[2]   # usually 640

        # Letterbox resize
        img = cv2.resize(frame, (target, target))
        img = img[:, :, ::-1].transpose(2, 0, 1)   # BGR→RGB, HWC→CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)                # [1,3,H,W]

        outputs = self._session.run(None, {self._input_name: img})[0]
        # outputs shape: [1, 84, 8400] for COCO — [1, nc+4, 8400] for ours
        # Transpose to [8400, nc+4]
        preds = outputs[0].T

        dets = []
        for pred in preds:
            x, y, w, h = pred[:4]
            class_scores = pred[4:]
            cls_id = int(np.argmax(class_scores))
            conf   = float(class_scores[cls_id])
            if conf < self.conf or cls_id >= len(CLASS_NAMES):
                continue

            # Convert from (cx, cy, w, h) normalized to pixel xyxy
            scale_x = w_orig / target
            scale_y = h_orig / target
            x1 = int((x - w / 2) * scale_x)
            y1 = int((y - h / 2) * scale_y)
            x2 = int((x + w / 2) * scale_x)
            y2 = int((y + h / 2) * scale_y)

            dets.append(RawDetection(
                bbox=[max(0, x1), max(0, y1), min(w_orig, x2), min(h_orig, y2)],
                confidence=conf,
                class_id=cls_id,
                vehicle_class=CLASS_NAMES[cls_id],
            ))

        return self._nms(dets)

    def _nms(self, dets: list[RawDetection]) -> list[RawDetection]:
        """Simple NMS for ONNX path (Ultralytics does this internally)."""
        if not dets:
            return dets
        import cv2
        boxes  = np.array([d.bbox for d in dets], dtype=np.float32)
        scores = np.array([d.confidence for d in dets], dtype=np.float32)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.conf, self.iou
        )
        return [dets[i] for i in (indices.flatten() if len(indices) else [])]

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def avg_fps(self) -> float:
        return sum(self._fps_log) / len(self._fps_log) if self._fps_log else 0.0
