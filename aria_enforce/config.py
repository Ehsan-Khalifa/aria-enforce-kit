"""
ARIA-Enforce — Violation catalog, fine schedule and scoring weights.

Single source of truth for: what violations exist, how they map to the
Motor Vehicles Act (Amendment) 2019, the fine amount, and the base
severity weight used for risk ranking.

Fine amounts are indicative (MV Act 2019, base values; states vary).
They are configurable here so a deploying authority can localise them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ViolationSpec:
    code:          str          # internal key, e.g. "no_helmet"
    label:         str          # human label for evidence / challan
    mv_act_section: str         # legal reference
    fine_inr:      int          # base fine in INR
    severity_base: int          # 1 (minor) .. 5 (grave) — danger-to-life weight
    detector:      str          # "rider_model" | "geometry" | "vehicle_model" | "flow"
    needs_plate:   bool = True  # whether a challan needs the plate


# ──────────────────────────────────────────────────────────────────────────────
# VIOLATION CATALOG — covers all 7 required violation types + existing ones.
# severity_base reflects danger to human life, not just legality.
# ──────────────────────────────────────────────────────────────────────────────
VIOLATION_CATALOG: dict[str, ViolationSpec] = {
    # ── Rider-attribute model (helmet/seatbelt/triple) ───────────────────────
    "no_helmet": ViolationSpec(
        code="no_helmet", label="Helmet non-compliance",
        mv_act_section="§194D MV Act", fine_inr=1000, severity_base=5,
        detector="rider_model"),
    "no_seatbelt": ViolationSpec(
        code="no_seatbelt", label="Seatbelt non-compliance",
        mv_act_section="§194B MV Act", fine_inr=1000, severity_base=4,
        detector="rider_model"),
    "triple_riding": ViolationSpec(
        code="triple_riding", label="Triple riding (overloading 2-wheeler)",
        mv_act_section="§194A MV Act", fine_inr=1000, severity_base=4,
        detector="rider_model"),

    # ── Geometry / scene rules ───────────────────────────────────────────────
    "stop_line": ViolationSpec(
        code="stop_line", label="Stop-line violation",
        mv_act_section="§184 MV Act", fine_inr=500, severity_base=2,
        detector="geometry"),
    "red_light_run": ViolationSpec(
        code="red_light_run", label="Red-light violation",
        mv_act_section="§184 MV Act", fine_inr=1000, severity_base=5,
        detector="geometry"),
    "illegal_parking": ViolationSpec(
        code="illegal_parking", label="Illegal / no-zone parking",
        mv_act_section="§122/177 MV Act", fine_inr=500, severity_base=2,
        detector="geometry"),
    "wrong_way": ViolationSpec(
        code="wrong_way", label="Wrong-side / wrong-way driving",
        mv_act_section="§184 MV Act", fine_inr=1100, severity_base=5,
        detector="flow"),

    # ── Vehicle-model / kinematics (carried over from live pipeline) ─────────
    "speeding": ViolationSpec(
        code="speeding", label="Over-speeding",
        mv_act_section="§183 MV Act", fine_inr=1000, severity_base=4,
        detector="vehicle_model"),
    "heavy_vehicle_peak_hour": ViolationSpec(
        code="heavy_vehicle_peak_hour", label="Heavy vehicle in peak-hour ban",
        mv_act_section="§194 MV Act", fine_inr=2000, severity_base=3,
        detector="geometry", needs_plate=True),

    # ── Pedestrian safety (cherry: uses existing pedestrian class) ───────────
    "pedestrian_crossing_block": ViolationSpec(
        code="pedestrian_crossing_block",
        label="Stopping on pedestrian crossing",
        mv_act_section="§184 MV Act", fine_inr=500, severity_base=3,
        detector="geometry", needs_plate=True),
}


# Map the legacy live-pipeline violation_type strings → catalog codes,
# so the existing ViolationDetector output plugs straight into challan/severity.
LEGACY_TYPE_ALIAS = {
    "red_light_run":            "red_light_run",
    "wrong_way":                "wrong_way",
    "speeding":                 "speeding",
    "heavy_vehicle_peak_hour":  "heavy_vehicle_peak_hour",
}


# ──────────────────────────────────────────────────────────────────────────────
# HELMET model — trained on datasets/helmet_rider (gw-khadatkar v5).
# Raw class names come straight from that data.yaml; we map them to internal
# semantics so the rest of the code never hard-codes Roboflow's labels.
# ──────────────────────────────────────────────────────────────────────────────
HELMET_MODEL_CLASSES = ["With Helmet", "Without Helmet", "licence"]

# raw model label  →  internal semantic token used by image_violations
CLASS_SEMANTICS = {
    "With Helmet":    "helmet",
    "Without Helmet": "no_helmet",
    "licence":        "number_plate",
    # seatbelt model (best-effort / pretrained) tokens:
    "Seatbelt":       "seatbelt",
    "No Seatbelt":    "no_seatbelt",
    "no_seatbelt":    "no_seatbelt",
    "seatbelt":       "seatbelt",
}

def semantic(raw_label: str) -> str:
    return CLASS_SEMANTICS.get(raw_label, raw_label)

# Default model weight locations (swapped in after training).
HELMET_MODEL_WEIGHTS = "models/helmet/weights/best.pt"
RIDER_MODEL_WEIGHTS = HELMET_MODEL_WEIGHTS   # backwards-compat alias
RIDER_MODEL_CLASSES = HELMET_MODEL_CLASSES   # backwards-compat alias
SEATBELT_MODEL_WEIGHTS = "models/seatbelt/best.pt"
VEHICLE_MODEL_WEIGHTS = (
    "models/vehicle/best.pt"
)


# ──────────────────────────────────────────────────────────────────────────────
# Severity scoring weights. final = base * density_factor * repeat_factor,
# clamped to 1..5 then bucketed.
# ─────────────�
SEVERITY_BUCKETS = {
    (0.0, 2.0): "LOW",
    (2.0, 3.5): "MEDIUM",
    (3.5, 4.5): "HIGH",
    (4.5, 99.0): "CRITICAL",
}
REPEAT_OFFENDER_MULTIPLIER = 1.25

def get_spec(code):
    return VIOLATION_CATALOG.get(LEGACY_TYPE_ALIAS.get(code, code))
