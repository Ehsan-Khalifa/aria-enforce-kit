"""
ARIA-Enforce — Evidence & challan store (SQLite).

Standalone persistence so the demo needs no TimescaleDB/Docker. Stores every
issued challan + a pointer to its annotated evidence image, and answers the
analytics/reporting questions (task 7): counts by type, trends, search,
daily summary, repeat-offender lookup.
"""
from __future__ import annotations
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS challans (
    challan_id        TEXT PRIMARY KEY,
    issued_at         TEXT NOT NULL,
    intersection_id   TEXT,
    violation_code    TEXT,
    violation_label   TEXT,
    mv_act_section    TEXT,
    fine_inr          INTEGER,
    anonymized_plate  TEXT,
    vehicle_class     TEXT,
    confidence        REAL,
    severity_score    REAL,
    severity_bucket   TEXT,
    evidence_image    TEXT,
    status            TEXT DEFAULT 'ISSUED'
);
CREATE INDEX IF NOT EXISTS idx_plate ON challans(anonymized_plate);
CREATE INDEX IF NOT EXISTS idx_code  ON challans(violation_code);
CREATE INDEX IF NOT EXISTS idx_date  ON challans(issued_at);
"""


class EvidenceStore:
    def __init__(self, db_path: str = "enforce_evidence.db"):
        self.db_path = db_path
        with closing(self._conn()) as c:
            c.executescript(_SCHEMA)
            c.commit()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── writes ────────────────────────────────────────────────────────────────
    def save(self, challan: dict) -> None:
        cols = ["challan_id", "issued_at", "intersection_id", "violation_code",
                "violation_label", "mv_act_section", "fine_inr",
                "anonymized_plate", "vehicle_class", "confidence",
                "severity_score", "severity_bucket", "evidence_image", "status"]
        vals = [challan.get(k) for k in cols]
        with closing(self._conn()) as c:
            c.execute(f"INSERT OR REPLACE INTO challans ({','.join(cols)}) "
                      f"VALUES ({','.join('?' * len(cols))})", vals)
            c.commit()

    def is_repeat_offender(self, plate: str) -> bool:
        if not plate or plate in ("UNKNOWN", "INVALID"):
            return False
        with closing(self._conn()) as c:
            n = c.execute("SELECT COUNT(*) FROM challans WHERE anonymized_plate=?",
                          (plate,)).fetchone()[0]
        return n > 0

    # ── analytics / reporting (task 7) ──────────────────────────────────────────
    def counts_by_type(self) -> list[dict]:
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT violation_label, COUNT(*) n, SUM(fine_inr) total_fine "
                "FROM challans GROUP BY violation_code ORDER BY n DESC").fetchall()
        return [dict(r) for r in rows]

    def daily_summary(self, day: Optional[str] = None) -> dict:
        day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with closing(self._conn()) as c:
            row = c.execute(
                "SELECT COUNT(*) challans, COALESCE(SUM(fine_inr),0) revenue, "
                "COALESCE(AVG(severity_score),0) avg_sev "
                "FROM challans WHERE substr(issued_at,1,10)=?", (day,)).fetchone()
        return {"date": day, **dict(row)}

    def search(self, plate: Optional[str] = None, code: Optional[str] = None,
               bucket: Optional[str] = None, limit: int = 100) -> list[dict]:
        q = "SELECT * FROM challans WHERE 1=1"
        args = []
        if plate:  q += " AND anonymized_plate LIKE ?"; args.append(f"%{plate}%")
        if code:   q += " AND violation_code=?"; args.append(code)
        if bucket: q += " AND severity_bucket=?"; args.append(bucket)
        q += " ORDER BY issued_at DESC LIMIT ?"; args.append(limit)
        with closing(self._conn()) as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def trend(self) -> list[dict]:
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT substr(issued_at,1,10) day, COUNT(*) n "
                "FROM challans GROUP BY day ORDER BY day").fetchall()
        return [dict(r) for r in rows]
