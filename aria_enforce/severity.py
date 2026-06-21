"""
ARIA-Enforce — Severity / risk scoring.

Turns a raw violation into a prioritised risk score so enforcement
resources go to the most dangerous offences first (this is the
"more than they asked" enforcement-intelligence layer).

score = base_severity * detection_confidence_factor
              * scene_density_factor * repeat_offender_factor
clamped to [1, 5] and bucketed to LOW / MEDIUM / HIGH / CRITICAL.
"""

from __future__ import annotations
from dataclasses import dataclass

from .config import get_spec, SEVERITY_BUCKETS, REPEAT_OFFENDER_MULTIPLIER


@dataclass
class SeverityResult:
    score:  float      # 1.0 .. 5.0
    bucket: str        # LOW | MEDIUM | HIGH | CRITICAL
    rationale: str


def _bucket(score: float) -> str:
    for (lo, hi), name in SEVERITY_BUCKETS.items():
        if lo <= score < hi:
            return name
    return "LOW"


def score_violation(violation_code: str,
                    confidence: float = 1.0,
                    scene_vehicle_count: int = 1,
                    is_repeat_offender: bool = False) -> SeverityResult:
    """
    Args:
        violation_code:       catalog code, e.g. "no_helmet"
        confidence:           model/rule confidence 0..1
        scene_vehicle_count:  vehicles in the frame (proxy for congestion risk)
        is_repeat_offender:   plate seen offending before
    """
    spec = get_spec(violation_code)
    base = spec.severity_base if spec else 2

    # Confidence dampens score for shaky detections (0.5..1.0 range).
    conf_factor = 0.5 + 0.5 * max(0.0, min(1.0, confidence))

    # A dangerous act in dense traffic endangers more people.
    if scene_vehicle_count >= 12:
        density_factor = 1.15
    elif scene_vehicle_count >= 6:
        density_factor = 1.05
    else:
        density_factor = 1.0

    repeat_factor = REPEAT_OFFENDER_MULTIPLIER if is_repeat_offender else 1.0

    raw = base * conf_factor * density_factor * repeat_factor
    score = round(max(1.0, min(5.0, raw)), 2)

    bits = [f"base={base}", f"conf×{conf_factor:.2f}"]
    if density_factor > 1.0:
        bits.append(f"density×{density_factor:.2f}")
    if is_repeat_offender:
        bits.append("repeat-offender")
    return SeverityResult(score=score, bucket=_bucket(score),
                          rationale=", ".join(bits))
