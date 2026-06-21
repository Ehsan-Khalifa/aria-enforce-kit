from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .rider_model import RiderInference, RiderBox
from .config import semantic

def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0: return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + ab - inter + 1e-6)

def _contains_centre(box, outer):
    cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]

def _point_in_poly(x, y, poly):
    inside = False; n = len(poly); j = n-1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-9)+xi):
            inside = not inside
        j = i
    return inside

@dataclass
class ImageViolation:
    code: str; bbox: list; confidence: float
    vehicle_class: str = "unknown"; plate: str = "UNKNOWN"; note: str = ""

def detect_rider_violations(rider, two_wheelers=None):
    out = []
    if rider.status != "ok" or not rider.boxes: return out
    helmets = [b for b in rider.boxes if semantic(b.kind) == "helmet"]
    bareheads = [b for b in rider.boxes if semantic(b.kind) == "no_helmet"]
    noseatbelt = [b for b in rider.boxes if semantic(b.kind) == "no_seatbelt"]
    for h in bareheads:
        out.append(ImageViolation("no_helmet", h.bbox, h.confidence, "two_wheeler"))
    for s in noseatbelt:
        out.append(ImageViolation("no_seatbelt", s.bbox, s.confidence, "car_suv"))
    heads = helmets + bareheads
    for tw in (two_wheelers or []):
        if tw.get("vehicle_class") != "two_wheeler": continue
        b = tw["bbox"]; env = [b[0]-20, b[1]-60, b[2]+20, b[3]+10]
        n = sum(1 for hh in heads if _contains_centre(hh.bbox, env))
        if n >= 3:
            out.append(ImageViolation("triple_riding", b, float(tw.get("confidence",0.7)),
                                      "two_wheeler", note=f"{n} occupants detected"))
    return out

def detect_zone_violations(vehicles, zones=None, signal_state=None):
    out = []
    if not zones: return out
    stop_y = zones.get("stop_line_y"); parking = zones.get("no_parking"); crossing = zones.get("crossing")
    for v in vehicles:
        bbox = v["bbox"]; conf = float(v.get("confidence",0.5)); vcls = v.get("vehicle_class","unknown")
        front_y = bbox[3]; cx, cy = (bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2
        if stop_y is not None and front_y > stop_y:
            if signal_state == "RED":
                out.append(ImageViolation("red_light_run", bbox, conf, vcls))
            else:
                out.append(ImageViolation("stop_line", bbox, conf, vcls, note="front past stop-line"))
        if parking and _point_in_poly(cx, cy, [tuple(p) for p in parking]):
            out.append(ImageViolation("illegal_parking", bbox, conf, vcls, note="vehicle in no-parking zone"))
        if crossing and _point_in_poly(cx, cy, [tuple(p) for p in crossing]):
            out.append(ImageViolation("pedestrian_crossing_block", bbox, conf, vcls, note="vehicle on crossing"))
    return out
