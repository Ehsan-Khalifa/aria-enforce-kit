"""
ARIA-Enforce — Single-image enforcement pipeline (the demo's engine).

Flow per uploaded image:
    image -> weather enhance -> vehicle detect -> helmet (adaptive) ->
    violations -> plate OCR -> severity -> e-challan -> store -> annotate.

Every heavy dependency is imported lazily and degrades gracefully.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .config import (VEHICLE_MODEL_WEIGHTS, HELMET_MODEL_WEIGHTS, get_spec,
                     semantic)
from .rider_model import RiderAttributeModel, RiderBox, RiderInference
from .image_violations import (detect_rider_violations, detect_zone_violations,
                               ImageViolation)
from .severity import score_violation
from .challan import issue_challan, Challan
from .store import EvidenceStore
from .annotate import annotate

logger = logging.getLogger("aria.enforce.pipeline")


@dataclass
class EnforceResult:
    annotated:   np.ndarray
    challans:    list = field(default_factory=list)
    violations:  list = field(default_factory=list)
    vehicles:    list = field(default_factory=list)
    weather:     str = "n/a"
    status:      dict = field(default_factory=dict)


class EnforcePipeline:
    def __init__(self,
                 vehicle_weights: str = VEHICLE_MODEL_WEIGHTS,
                 helmet_weights: str = HELMET_MODEL_WEIGHTS,
                 db_path: str = "enforce_evidence.db",
                 evidence_dir: str = "evidence",
                 intersection_id: str = "RJK-C-001",
                 conf: float = 0.4,
                 device: str = "auto"):
        self.intersection_id = intersection_id
        self.conf = conf
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.store = EvidenceStore(db_path)
        self.helmet = RiderAttributeModel(helmet_weights, conf, device)
        self._vehicle = self._load_vehicle(vehicle_weights, conf, device)
        self._ocr = self._load_ocr()
        self._weather = self._load_weather()

    # ── lazy loaders (graceful) ────────────────────────────────────────────────
    def _load_vehicle(self, weights, conf, device):
        try:
            from aria_enforce.vision.detector import VehicleDetector
            if not Path(weights).exists():
                logger.warning(f"[pipeline] vehicle weights missing: {weights}")
                return None
            return VehicleDetector(weights, conf_threshold=conf, device=device)
        except Exception as e:
            logger.warning(f"[pipeline] vehicle detector unavailable: {e}")
            return None

    def _load_ocr(self):
        try:
            from aria_enforce.vision.ocr import PlateReader
            return PlateReader()
        except Exception as e:
            logger.warning(f"[pipeline] OCR unavailable: {e}")
            return None

    def _load_weather(self):
        try:
            from aria_enforce.vision.weather import WeatherProcessor
            return WeatherProcessor()
        except Exception as e:
            logger.warning(f"[pipeline] weather processor unavailable: {e}")
            return None

    @property
    def status(self) -> dict:
        return {
            "vehicle_model": "ok" if self._vehicle else "missing",
            "helmet_model":  self.helmet._status,
            "ocr":           "ok" if self._ocr else "missing",
            "weather":       "ok" if self._weather else "missing",
        }

    # ── helpers ────────────────────────────────────────────────────────────────
    def _detect_vehicles(self, frame) -> list[dict]:
        if not self._vehicle:
            return []
        dets = self._vehicle.detect(frame)
        return [{"bbox": d.bbox, "vehicle_class": d.vehicle_class,
                 "confidence": d.confidence} for d in dets]

    def _helmet_riders(self, frame, two_wheelers) -> RiderInference:
        """Adaptive helmet stage. If the helmet model genuinely LOCALIZES
        (a scene detector returns multiple non-full-frame boxes), use those
        directly. If it behaves like a whole-image CLASSIFIER, fall back to
        per-rider crops. Works before and after the detection-dataset retrain."""
        if not self.helmet.ready:
            return RiderInference(status=self.helmet._status, boxes=[])
        full = self.helmet.infer(frame)
        if full.status == "ok" and full.boxes:
            h, w = frame.shape[:2]
            area = float(max(1, h * w))
            proper = []
            for b in full.boxes:
                if semantic(b.kind) not in ("helmet", "no_helmet"):
                    continue
                bw = max(0, b.bbox[2] - b.bbox[0])
                bh = max(0, b.bbox[3] - b.bbox[1])
                if (bw * bh) < 0.55 * area and b.confidence >= 0.45:
                    proper.append(b)
            if proper:
                return RiderInference(status="ok", boxes=proper)
        return self._classify_riders(frame, two_wheelers)

    def _classify_riders(self, frame, two_wheelers) -> RiderInference:
        """Two-stage helmet check: classify each two-wheeler rider crop and keep
        the single winning verdict (helmet vs no_helmet) per rider."""
        if not self.helmet.ready:
            return RiderInference(status=self.helmet._status, boxes=[])
        if not two_wheelers:
            return RiderInference(status="ok", boxes=[])
        h, w = frame.shape[:2]
        boxes: list[RiderBox] = []
        for tw in two_wheelers:
            x1, y1, x2, y2 = [int(c) for c in tw["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            ey1 = max(0, y1 - int(0.6 * (y2 - y1)))   # extend up to include the head
            crop = frame[ey1:y2, x1:x2]
            if crop.size == 0:
                continue
            res = self.helmet.infer(crop)
            if res.status != "ok" or not res.boxes:
                continue
            cand = [b for b in res.boxes
                    if semantic(b.kind) in ("helmet", "no_helmet")]
            if not cand:
                continue
            best = max(cand, key=lambda b: b.confidence)
            if best.confidence < 0.50:        # drop low-confidence/uncertain verdicts
                continue
            fb = [best.bbox[0] + x1, best.bbox[1] + ey1,
                  best.bbox[2] + x1, best.bbox[3] + ey1]
            boxes.append(RiderBox(bbox=fb, kind=best.kind,
                                  confidence=best.confidence))
        return RiderInference(status="ok", boxes=boxes)

    def _ocr_plate(self, frame, bbox) -> str:
        if not self._ocr:
            return "UNKNOWN"
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return "UNKNOWN"
        roi = frame[y1:y2, x1:x2]
        try:
            return self._ocr.read_and_anonymize(roi).get("anonymized", "UNKNOWN")
        except Exception:
            return "UNKNOWN"

    # ── main entry ─────────────────────────────────────────────────────────────
    def process(self, img_bgr: np.ndarray,
                signal_state: Optional[str] = None,
                zones: Optional[dict] = None,
                issue: bool = True) -> EnforceResult:
        # 1. weather assess + enhance
        if self._weather:
            proc, wres = self._weather.process(img_bgr)
            weather = wres.condition.value
        else:
            proc, weather = img_bgr, "n/a"

        # 2. vehicles
        vehicles = self._detect_vehicles(proc)
        two_wheelers = [v for v in vehicles if v.get("vehicle_class") == "two_wheeler"]

        # 3. rider attributes (adaptive: detector boxes, else per-rider crops)
        rider = self._helmet_riders(proc, two_wheelers)

        # 4. violations
        vlist: list[ImageViolation] = []
        vlist += detect_rider_violations(rider, two_wheelers)
        vlist += detect_zone_violations(vehicles, zones, signal_state)

        scene_n = max(1, len(vehicles))
        challans: list[Challan] = []
        enriched: list[dict] = []

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ev_path = str(self.evidence_dir / f"evidence_{self.intersection_id}_{ts}.jpg")

        # 5-6. plate OCR → severity → challan → store
        for v in vlist:
            spec = get_spec(v.code)
            plate = self._ocr_plate(proc, v.bbox) if (spec and spec.needs_plate) else "N/A"
            repeat = self.store.is_repeat_offender(plate)
            sev = score_violation(v.code, v.confidence, scene_n, repeat)
            ch = issue_challan(self.intersection_id, v.code, plate,
                               v.vehicle_class, v.confidence,
                               evidence_image=ev_path,
                               scene_vehicle_count=scene_n,
                               is_repeat_offender=repeat, severity=sev)
            if issue:
                self.store.save(ch.to_dict())
            challans.append(ch)
            enriched.append({"bbox": v.bbox, "code": v.code,
                             "confidence": v.confidence,
                             "severity_bucket": sev.bucket, "plate": plate,
                             "note": v.note})

        # 7. annotate + persist evidence
        annotated = annotate(proc, enriched, self.intersection_id, weather)
        try:
            import cv2
            cv2.imwrite(ev_path, annotated)
        except Exception as e:
            logger.warning(f"[pipeline] could not write evidence: {e}")

        return EnforceResult(annotated=annotated, challans=challans,
                             violations=enriched, vehicles=vehicles,
                             weather=weather, status=self.status)
