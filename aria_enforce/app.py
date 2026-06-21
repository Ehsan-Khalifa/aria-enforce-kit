"""
ARIA-Enforce — Demo app (Gradio).

One-command, no-Docker reviewer surface for the Round-2 submission:
  Tab 1  Violation Detection  — upload a traffic photo → annotated evidence,
                                detected violations, severity, plate, e-challans.
  Tab 2  Analytics & Reports  — counts by type, trend, daily revenue, search.
  Tab 3  Event & Congestion   — historical hotspots + pre-emptive risk alerts.

Run:
    pip install -r aria/enforce/requirements-enforce.txt
    python -m aria.enforce.app          # from project root
"""
from __future__ import annotations
import os
import datetime as _dt

import cv2
import numpy as np
import pandas as pd
import gradio as gr

from aria_enforce.image_pipeline import EnforcePipeline
from aria_enforce.challan import challan_text
from aria_enforce import event_alert as ea

DB_PATH = os.environ.get("ENFORCE_DB", "enforce_evidence.db")
EVENT_CSV = os.environ.get("ENFORCE_EVENTS", "data/astram_events.csv")

_pipeline: EnforcePipeline | None = None
_events_df = None


def pipeline() -> EnforcePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = EnforcePipeline(db_path=DB_PATH)
    return _pipeline


def events_df():
    global _events_df
    if _events_df is None and os.path.exists(EVENT_CSV):
        _events_df = ea.load_events(EVENT_CSV)
    return _events_df


# ── Tab 1: detection ────────────────────────────────────────────────────────
def run_detection(image, signal_state, stop_line_y):
    if image is None:
        return None, [], "Upload an image to begin."
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    zones = {"stop_line_y": int(stop_line_y)} if stop_line_y else None
    sig = None if signal_state == "Unknown" else signal_state.upper()
    res = pipeline().process(bgr, signal_state=sig, zones=zones)

    rgb = cv2.cvtColor(res.annotated, cv2.COLOR_BGR2RGB)
    rows = [[c.challan_id, c.violation_label, f"{c.confidence:.0%}",
             c.severity_bucket, f"Rs {c.fine_inr}", c.anonymized_plate]
            for c in res.challans]
    if res.challans:
        slips = "\n\n".join(challan_text(c) for c in res.challans)
    else:
        st = res.status
        slips = ("No violations detected.\n\nModel status: "
                 f"vehicle={st['vehicle_model']}, helmet={st['helmet_model']}, "
                 f"ocr={st['ocr']}, weather={st['weather']}")
    return rgb, rows, slips


# ── Tab 2: analytics ────────────────────────────────────────────────────────
def load_analytics():
    s = pipeline().store
    by_type = [[r["violation_label"], r["n"], f"Rs {r['total_fine'] or 0}"]
               for r in s.counts_by_type()]
    today = s.daily_summary()
    summary = (f"Today ({today['date']}): {today['challans']} challans · "
               f"Rs {today['revenue']} fines · avg severity {today['avg_sev']:.2f}/5")
    trend = [[r["day"], r["n"]] for r in s.trend()]
    return by_type, summary, trend


def search_challans(plate, bucket):
    s = pipeline().store
    rows = s.search(plate=plate or None,
                    bucket=(None if bucket == "Any" else bucket))
    return [[r["challan_id"], r["violation_label"], r["severity_bucket"],
             f"Rs {r['fine_inr']}", r["anonymized_plate"], r["issued_at"][:16]]
            for r in rows]


# ── Tab 3: event/congestion ─────────────────────────────────────────────────
def load_events_view(by, hour):
    df = events_df()
    if df is None:
        return [], [], "Event data not found at " + EVENT_CSV
    hs = [[h.name, h.events, h.top_cause, f"{h.closure_rate:.0%}"]
          for h in ea.hotspots(df, by, 10)]
    alerts = [[a["location"], a["risk"], a["band"]]
              for a in ea.current_alerts(df, int(hour), by)]
    note = (f"{len(df)} historical events · "
            f"{df['event_type'].value_counts().to_dict()}")
    return hs, alerts, note


# ── UI ──────────────────────────────────────────────────────────────────────
def build():
    with gr.Blocks(title="ARIA-Enforce") as demo:
        gr.Markdown("# 🚦 ARIA-Enforce — AI Traffic Violation Detection\n"
                    "Upload a traffic photo → detect violations → auto e-challan "
                    "→ analytics. Part of the ITMS adaptive-traffic platform.")
        with gr.Tab("1 · Violation Detection"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(label="Traffic image", type="numpy")
                    sig = gr.Dropdown(["Unknown", "Red", "Green", "Amber"],
                                      value="Unknown", label="Signal state")
                    sly = gr.Slider(0, 2000, value=0, step=10,
                                    label="Stop-line y-pixel (0 = off)")
                    btn = gr.Button("Detect violations", variant="primary")
                with gr.Column():
                    out_img = gr.Image(label="Annotated evidence")
            out_tbl = gr.Dataframe(
                headers=["Challan", "Violation", "Conf", "Severity", "Fine", "Plate"],
                label="Issued e-challans")
            out_txt = gr.Textbox(label="e-Challan slips", lines=14)
            btn.click(run_detection, [inp, sig, sly], [out_img, out_tbl, out_txt])

        with gr.Tab("2 · Analytics & Reports"):
            a_btn = gr.Button("Refresh analytics")
            a_sum = gr.Textbox(label="Daily summary")
            a_tbl = gr.Dataframe(headers=["Violation", "Count", "Fines"],
                                 label="Violations by type")
            a_trd = gr.Dataframe(headers=["Day", "Challans"], label="Daily trend")
            a_btn.click(load_analytics, None, [a_tbl, a_sum, a_trd])
            gr.Markdown("### Search records")
            with gr.Row():
                s_plate = gr.Textbox(label="Plate contains")
                s_bucket = gr.Dropdown(["Any", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
                                       value="Any", label="Severity")
                s_btn = gr.Button("Search")
            s_tbl = gr.Dataframe(
                headers=["Challan", "Violation", "Severity", "Fine", "Plate", "When"])
            s_btn.click(search_challans, [s_plate, s_bucket], s_tbl)

        with gr.Tab("3 · Event & Congestion Alerts"):
            gr.Markdown("Predictive layer over historical traffic-event data.")
            with gr.Row():
                e_by = gr.Dropdown(["junction", "zone", "corridor"],
                                   value="junction", label="Group by")
                e_hr = gr.Slider(0, 23, value=18, step=1, label="Hour of day")
                e_btn = gr.Button("Analyze")
            e_note = gr.Textbox(label="Dataset")
            e_hs = gr.Dataframe(headers=["Location", "Events", "Top cause", "Closure%"],
                                label="Congestion hotspots")
            e_al = gr.Dataframe(headers=["Location", "Risk", "Band"],
                                label="Pre-emptive alerts for this hour")
            e_btn.click(load_events_view, [e_by, e_hr], [e_hs, e_al, e_note])
    return demo


if __name__ == "__main__":
    build().launch()
