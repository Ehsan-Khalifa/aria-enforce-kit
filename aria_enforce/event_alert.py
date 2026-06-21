"""
ARIA-Enforce — Event & Congestion Alert (the "more than they asked" panel).

A lightweight forecaster over historical traffic-event data (Astram).
It does NOT pretend to be a full congestion model — it mines history to:
  * rank congestion HOTSPOTS (junction / zone / corridor),
  * estimate a RISK SCORE for a location at a given hour/day from how often
    events have struck there historically, and
  * emit pre-emptive ALERTS for the current time window.

This demonstrates the platform vision (enforcement -> prediction) for the
cost of a small, honest, data-driven module.
"""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Optional

_NULLS = {"NULL", "", "nan", "None", None}


def _clean(s):
    return None if (isinstance(s, str) and s.strip() in _NULLS) else s


def load_events(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["start_dt"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    df["end_dt"]   = pd.to_datetime(df["end_datetime"], errors="coerce", utc=True)
    df["hour"]     = df["start_dt"].dt.hour
    df["dow"]      = df["start_dt"].dt.dayofweek          # 0=Mon
    df["duration_min"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60
    for c in ("junction", "zone", "corridor", "event_type", "event_cause",
              "priority", "requires_road_closure"):
        if c in df.columns:
            df[c] = df[c].map(_clean)
    return df


@dataclass
class Hotspot:
    name: str
    events: int
    top_cause: str
    closure_rate: float          # fraction needing road closure
    avg_duration_min: Optional[float]


def hotspots(df: pd.DataFrame, by: str = "junction", top: int = 10) -> list[Hotspot]:
    sub = df[df[by].notna()]
    out = []
    for name, g in sub.groupby(by):
        cause = g["event_cause"].dropna()
        closure = g["requires_road_closure"].astype(str).str.upper().eq("TRUE").mean()
        dur = g["duration_min"]
        dur = float(dur[dur.between(0, 60 * 24)].mean()) if dur.notna().any() else None
        out.append(Hotspot(
            name=str(name), events=len(g),
            top_cause=(cause.mode().iat[0] if not cause.empty else "unknown"),
            closure_rate=round(float(closure), 3),
            avg_duration_min=round(dur, 1) if dur and dur == dur else None))
    out.sort(key=lambda h: h.events, reverse=True)
    return out[:top]


def risk_score(df: pd.DataFrame, location: str, hour: int,
               by: str = "junction") -> dict:
    """0..100 risk that a congestion event hits `location` around `hour`."""
    sub = df[df[by] == location]
    if sub.empty:
        return {"location": location, "risk": 0, "band": "UNKNOWN", "basis": 0}
    total = len(sub)
    window = sub[sub["hour"].between((hour - 1) % 24, (hour + 1) % 24)]
    base = total / max(1, df[by].notna().sum())             # prevalence
    timed = len(window) / max(1, total)                      # time concentration
    closure = sub["requires_road_closure"].astype(str).str.upper().eq("TRUE").mean()
    raw = 100 * (0.5 * min(1, base * 50) + 0.3 * timed + 0.2 * closure)
    risk = int(max(0, min(100, raw)))
    band = ("CRITICAL" if risk >= 70 else "HIGH" if risk >= 45
            else "MEDIUM" if risk >= 20 else "LOW")
    return {"location": location, "risk": risk, "band": band, "basis": total}


def current_alerts(df: pd.DataFrame, hour: int, by: str = "junction",
                   min_events: int = 15, top: int = 5) -> list[dict]:
    """Pre-emptive alerts: busiest locations whose history peaks near `hour`."""
    counts = df[df[by].notna()][by].value_counts()
    candidates = counts[counts >= min_events].index.tolist()
    scored = [risk_score(df, loc, hour, by) for loc in candidates]
    scored = [s for s in scored if s["band"] in ("HIGH", "CRITICAL")]
    scored.sort(key=lambda s: s["risk"], reverse=True)
    return scored[:top]


if __name__ == "__main__":
    import sys
    df = load_events(sys.argv[1])
    print("rows:", len(df), "| with junction:", df["junction"].notna().sum())
    print("planned/unplanned:", df["event_type"].value_counts().to_dict())
    print("\nTOP JUNCTION HOTSPOTS:")
    for h in hotspots(df, "junction", 5):
        print(f"  {h.name:30s} events={h.events:4d} cause={h.top_cause:20s} "
              f"closure={h.closure_rate:.0%} dur={h.avg_duration_min}")
    print("\nALERTS @ 18:00:")
    for a in current_alerts(df, 18, "junction"):
        print(" ", a)
