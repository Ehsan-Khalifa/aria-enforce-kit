"""
ARIA-Enforce — Demo app (Gradio).

Clean, tabular, one-command reviewer surface:
  Tab 1  Detect Violations   — upload a photo -> annotated evidence + a table
                               of violations and auto-generated e-challans.
  Tab 2  Analytics & Reports — totals, by-type table, daily summary, search.
  Tab 3  Congestion Forecast — historical hotspots + pre-emptive risk alerts.

Run:
    python -m aria_enforce.app            # local
    python -m aria_enforce.app --share    # public link (for the Demo Link field)
"""
from __future__ import annotations
import os
import sys

import cv2
import gradio as gr

from aria_enforce.image_pipeline import EnforcePipeline
from aria_enforce import event_alert as ea

DB_PATH = os.environ.get("ENFORCE_DB", "enforce_evidence.db")
EVENT_CSV = os.environ.get("ENFORCE_EVENTS", "data/astram_events.csv")

_pipeline = None
_events_df = None

VIOLATION_HEADERS = ["#", "Violation", "Severity", "Confidence",
                     "Vehicle", "Plate", "Fine (Rs)", "Challan ID"]


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


def _status_line() -> str:
    s = pipeline().status
    nice = {"ok": "ready", "missing": "not loaded",
            "model_pending": "not trained yet", "error": "error"}
    return (f"**System status** — vehicle detector: {nice.get(s['vehicle_model'])} · "
            f"helmet model: {nice.get(s['helmet_model'])} · "
            f"plate OCR: {nice.get(s['ocr'])} · "
            f"image enhancement: {nice.get(s['weather'])}")


# ── Tab 1: detection ─────────────────────────────────────────────────────────
def run_detection(image, use_rules, signal_state, stop_line_y):
    if image is None:
        return None, "Upload a traffic photo and click **Analyze image**.", []
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    zones = None
    sig = None
    if use_rules:
        sig = None if signal_state == "Unknown" else signal_state.upper()
        if stop_line_y and stop_line_y > 0:
            zones = {"stop_line_y": int(stop_line_y)}
    res = pipeline().process(bgr, signal_state=sig, zones=zones)
    rgb = cv2.cvtColor(res.annotated, cv2.COLOR_BGR2RGB)

    rows = []
    total = 0
    for i, c in enumerate(res.challans, 1):
        total += c.fine_inr
        rows.append([i, c.violation_label, c.severity_bucket,
                     f"{c.confidence:.0%}", c.vehicle_class, c.anonymized_plate,
                     c.fine_inr, c.challan_id])

    if rows:
        summary = (f"### {len(rows)} violation(s) detected — total fine Rs {total:,}\n"
                   f"Each row below is an auto-generated e-challan. "
                   f"Annotated evidence is on the right.")
    else:
        summary = ("### No violations detected\n"
                   "Either the scene is compliant, or no rider/vehicle was found. "
                   "Try a clearer photo of traffic.\n\n" + _status_line())
    return rgb, summary, rows


# ── Tab 2: analytics ─────────────────────────────────────────────────────────
def load_analytics():
    s = pipeline().store
    today = s.daily_summary()
    summary = (f"### Today — {today['challans']} challans · "
               f"Rs {today['revenue']:,} in fines · "
               f"avg severity {today['avg_sev']:.1f}/5")
    by_type = [[r["violation_label"], r["n"], f"Rs {r['total_fine'] or 0:,}"]
               for r in s.counts_by_type()]
    trend = [[r["day"], r["n"]] for r in s.trend()]
    if not by_type:
        summary = ("### No challans issued yet\n"
                   "Analyze some images in Tab 1 first — they'll appear here.")
    return summary, by_type, trend


def search_challans(plate, bucket):
    s = pipeline().store
    rows = s.search(plate=plate or None,
                    bucket=(None if bucket == "Any" else bucket))
    return [[r["challan_id"], r["violation_label"], r["severity_bucket"],
             f"Rs {r['fine_inr']:,}", r["anonymized_plate"], r["issued_at"][:16]]
            for r in rows]


# ── Tab 3: congestion ────────────────────────────────────────────────────────
def load_events_view(by, hour):
    df = events_df()
    if df is None:
        return ("Event data not found. Place the CSV at "
                f"`{EVENT_CSV}` to enable this tab.", [], [])
    counts = df["event_type"].value_counts().to_dict()
    note = (f"### {len(df):,} historical traffic events analysed\n"
            f"Planned vs unplanned: {counts}. "
            f"Hotspots and risk alerts for hour {int(hour):02d}:00 below.")
    hs = [[h.name, h.events, h.top_cause, f"{h.closure_rate:.0%}"]
          for h in ea.hotspots(df, by, 10)]
    alerts = [[a["location"], f"{a['risk']}/100", a["band"]]
              for a in ea.current_alerts(df, int(hour), by)]
    return note, hs, alerts


# ── UI ───────────────────────────────────────────────────────────────────────
def build():
    with gr.Blocks(title="ARIA-Enforce", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# ARIA-Enforce — AI Traffic Violation Detection\n"
            "Upload a traffic photo and the system detects road users, identifies "
            "violations (helmet, stop-line, red-light, illegal parking, …), reads "
            "the number plate, and issues an **e-challan** — with annotated "
            "evidence. Part of the ITMS adaptive-traffic platform.")

        with gr.Tab("1 · Detect Violations"):
            gr.Markdown("**How to use:** upload a photo → click **Analyze image** "
                        "→ review the violations table and annotated evidence.")
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(label="Traffic photo", type="numpy",
                                   height=300)
                    with gr.Accordion("Optional: intersection rules", open=False):
                        use_rules = gr.Checkbox(
                            label="Apply stop-line / red-light rules", value=False)
                        signal_state = gr.Dropdown(
                            ["Unknown", "Red", "Green", "Amber"],
                            value="Unknown", label="Signal state")
                        stop_line_y = gr.Slider(
                            0, 2000, value=0, step=10,
                            label="Stop-line position (y-pixel; 0 = off)")
                    btn = gr.Button("Analyze image", variant="primary", size="lg")
                with gr.Column(scale=1):
                    out_img = gr.Image(label="Annotated evidence", height=300)
            out_summary = gr.Markdown()
            out_tbl = gr.Dataframe(headers=VIOLATION_HEADERS, wrap=True,
                                   label="Violations & e-challans",
                                   interactive=False)
            btn.click(run_detection, [inp, use_rules, signal_state, stop_line_y],
                      [out_img, out_summary, out_tbl])

        with gr.Tab("2 · Analytics & Reports"):
            a_btn = gr.Button("Load / refresh analytics", variant="primary")
            a_sum = gr.Markdown()
            with gr.Row():
                a_tbl = gr.Dataframe(headers=["Violation", "Count", "Total fines"],
                                     label="Violations by type", interactive=False)
                a_trd = gr.Dataframe(headers=["Date", "Challans"],
                                     label="Daily trend", interactive=False)
            gr.Markdown("#### Search records")
            with gr.Row():
                s_plate = gr.Textbox(label="Plate contains", scale=2)
                s_bucket = gr.Dropdown(["Any", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
                                       value="Any", label="Severity", scale=1)
                s_btn = gr.Button("Search", scale=1)
            s_tbl = gr.Dataframe(
                headers=["Challan ID", "Violation", "Severity", "Fine",
                         "Plate", "Issued"], interactive=False)
            a_btn.click(load_analytics, None, [a_sum, a_tbl, a_trd])
            s_btn.click(search_challans, [s_plate, s_bucket], s_tbl)

        with gr.Tab("3 · Congestion Forecast"):
            gr.Markdown("Predictive layer over historical traffic-event data — "
                        "find congestion hotspots and get pre-emptive risk alerts.")
            with gr.Row():
                e_by = gr.Dropdown(["junction", "zone", "corridor"],
                                   value="junction", label="Group by", scale=1)
                e_hr = gr.Slider(0, 23, value=18, step=1,
                                 label="Hour of day", scale=2)
                e_btn = gr.Button("Analyze", variant="primary", scale=1)
            e_note = gr.Markdown()
            with gr.Row():
                e_hs = gr.Dataframe(
                    headers=["Location", "Events", "Top cause", "Road-closure rate"],
                    label="Congestion hotspots", interactive=False)
                e_al = gr.Dataframe(
                    headers=["Location", "Risk", "Band"],
                    label="Pre-emptive alerts for this hour", interactive=False)
            e_btn.click(load_events_view, [e_by, e_hr], [e_note, e_hs, e_al])
    return demo


if __name__ == "__main__":
    share = ("--share" in sys.argv) or os.environ.get("SHARE") == "1"
    build().launch(share=share)
