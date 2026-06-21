"""
ARIA-Enforce — Automated e-Challan generation.

Goes beyond detection into ENFORCEMENT: turns a confirmed violation into a
structured electronic challan (fine notice) with a unique id, legal section,
fine amount, anonymized plate, severity and a pointer to the annotated
evidence image. This is the headline "more than they asked" feature.

Plate handling: by default the challan stores the ANONYMIZED plate (privacy by
design, matching aria/src/ocr.py). A deploying RTO would resolve the real plate
behind an access-controlled boundary; we never persist raw plate text here.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from .config import get_spec
from .severity import SeverityResult, score_violation


@dataclass
class Challan:
    challan_id:       str
    issued_at:        str          # ISO timestamp
    intersection_id:  str
    violation_code:   str
    violation_label:  str
    mv_act_section:   str
    fine_inr:         int
    anonymized_plate: str
    vehicle_class:    str
    confidence:       float
    severity_score:   float
    severity_bucket:  str
    evidence_image:   Optional[str]    # path/URL to annotated evidence
    status:           str = "ISSUED"   # ISSUED | DISPUTED | PAID | VOID

    def to_dict(self) -> dict:
        return asdict(self)


def _gen_id(intersection_id: str, plate: str, code: str, ts: str) -> str:
    h = hashlib.sha1(f"{intersection_id}|{plate}|{code}|{ts}".encode()).hexdigest()[:8]
    date = ts[:10].replace("-", "")
    return f"CHL-{date}-{h.upper()}"


def issue_challan(intersection_id: str,
                  violation_code: str,
                  anonymized_plate: str,
                  vehicle_class: str,
                  confidence: float,
                  evidence_image: Optional[str] = None,
                  scene_vehicle_count: int = 1,
                  is_repeat_offender: bool = False,
                  severity: Optional[SeverityResult] = None) -> Challan:
    """Create a Challan from a confirmed violation."""
    spec = get_spec(violation_code)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if severity is None:
        severity = score_violation(violation_code, confidence,
                                   scene_vehicle_count, is_repeat_offender)

    return Challan(
        challan_id       = _gen_id(intersection_id, anonymized_plate,
                                   violation_code, ts),
        issued_at        = ts,
        intersection_id  = intersection_id,
        violation_code   = violation_code,
        violation_label  = spec.label if spec else violation_code,
        mv_act_section   = spec.mv_act_section if spec else "—",
        fine_inr         = spec.fine_inr if spec else 0,
        anonymized_plate = anonymized_plate or "UNKNOWN",
        vehicle_class    = vehicle_class,
        confidence       = round(confidence, 3),
        severity_score   = severity.score,
        severity_bucket  = severity.bucket,
        evidence_image   = evidence_image,
    )


def challan_text(c: Challan) -> str:
    """Render a printable challan slip (for the demo / PDF evidence)."""
    return (
        f"╔══════════════════════════════════════════════════════╗\n"
        f"║   e-CHALLAN  —  ARIA-Enforce Traffic Enforcement      ║\n"
        f"╠══════════════════════════════════════════════════════╣\n"
        f"  Challan No   : {c.challan_id}\n"
        f"  Issued       : {c.issued_at}\n"
        f"  Location     : {c.intersection_id}\n"
        f"  Vehicle      : {c.vehicle_class}   Plate: {c.anonymized_plate}\n"
        f"  Offence      : {c.violation_label}\n"
        f"  Legal        : {c.mv_act_section}\n"
        f"  Fine         : ₹ {c.fine_inr}\n"
        f"  Severity     : {c.severity_bucket}  ({c.severity_score}/5)\n"
        f"  Confidence   : {c.confidence:.0%}\n"
        f"  Evidence     : {c.evidence_image or 'attached'}\n"
        f"  Status       : {c.status}\n"
        f"╚══════════════════════════════════════════════════════╝"
    )
