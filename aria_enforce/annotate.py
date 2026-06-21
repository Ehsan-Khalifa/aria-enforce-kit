"""
ARIA-Enforce — Evidence annotation.

Draws violation boxes, labels, severity colour and a caption banner
(timestamp, location, plate) onto the image to produce court-ready
annotated evidence (task 6).
"""
from __future__ import annotations
import cv2
import numpy as np
from datetime import datetime, timezone

_BUCKET_COLOR = {
    "CRITICAL": (0, 0, 220), "HIGH": (0, 90, 230),
    "MEDIUM": (0, 190, 230), "LOW": (90, 170, 90), "UNKNOWN": (160, 160, 160),
}


def _put_label(img, text, x, y, color, scale=0.5):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(y, h + 6)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def annotate(img_bgr: np.ndarray, violations: list, intersection_id: str = "",
             weather: str = "") -> np.ndarray:
    """violations: list of dicts/objects with .bbox, .code, .confidence,
    .severity_bucket (optional), .plate (optional)."""
    out = img_bgr.copy()
    for v in violations:
        bbox = v["bbox"] if isinstance(v, dict) else v.bbox
        code = v["code"] if isinstance(v, dict) else v.code
        conf = (v.get("confidence") if isinstance(v, dict) else v.confidence) or 0
        bucket = (v.get("severity_bucket") if isinstance(v, dict)
                  else getattr(v, "severity_bucket", "MEDIUM")) or "MEDIUM"
        color = _BUCKET_COLOR.get(bucket, (0, 140, 230))
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _put_label(out, f"{code} {conf:.0%}", x1, y1, color)

    # caption banner
    h = out.shape[0]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    banner = f"ARIA-Enforce | {intersection_id} | {ts}"
    if weather:
        banner += f" | {weather}"
    banner += f" | {len(violations)} violation(s)"
    cv2.rectangle(out, (0, h - 24), (out.shape[1], h), (30, 30, 30), -1)
    cv2.putText(out, banner, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out
