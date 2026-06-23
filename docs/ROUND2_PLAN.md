# Round 2 Prototype — Build & Submission Plan
### Theme: Automated Photo Identification & Classification for Traffic Violations Using Computer Vision
**Project:** ITMS → repackaged as **"ARIA-Enforce"** (image-based traffic violation detection)
**Date:** 2026-06-21

---

## 1. Strategy — why we win

Most teams will submit a single notebook that runs YOLO on a few images. **We already have a production-grade, Indian-context vision stack** (trained YOLO26m on 11 Indian vehicle classes, multi-object tracking, plate OCR with anonymization, weather preprocessing, a violation engine, a backend, and a dashboard). Our edge is **depth, real Indian data, and a complete evidence→analytics pipeline**.

But to win we must do two things the current repo does NOT do:

1. **Reframe for the theme.** The problem statement is about *photographic* violation identification. Our system is a live-video + RL-signal pipeline. We expose an **image-first demo** that maps 1:1 onto the 8 required tasks.
2. **Close the violation gap.** We add the violations that judges expect for Indian roads and that we don't have yet: **helmet non-compliance, seatbelt, stop-line, illegal parking** (red-light + wrong-side already exist).
3. **Make it trivially runnable.** No Docker/TimescaleDB/Redis/MQTT for the reviewer — a one-command standalone app.

---

## 2. Gap analysis — 8 required tasks vs current repo

| # | Required task | Current state | Action |
|---|---------------|---------------|--------|
| 1 | Image preprocessing (low light, rain, shadow, blur) | `aria/src/weather.py` (CLAHE + scene assessment) ✅ | Wire into image path; add deblur/denoise toggle |
| 2 | Vehicle & road-user detection + categories | YOLO26m, 11 Indian classes, incl. pedestrian ✅ | Reuse as-is |
| 3 | Violation detection (helmet, seatbelt, triple, wrong-side, stop-line, red-light, illegal parking) | red-light, wrong-side, speeding, heavy-veh ✅ / helmet, seatbelt, stop-line, illegal parking ❌ | **Build:** rider-attribute model (helmet/seatbelt) + geometry rules (stop-line, parking) |
| 4 | Violation classification + confidence | `violation.py` returns type + confidence ✅ | Extend to new types |
| 5 | License plate recognition (OCR) | `aria/src/ocr.py` PaddleOCR + anonymization ✅ | Reuse; expose raw plate in demo mode (toggle) |
| 6 | Evidence generation (annotated images + metadata) | `publisher.py` sends to HERMES ⚠️ (no saved annotated image) | **Build:** annotated-image writer + SQLite metadata store |
| 7 | Analytics & reporting (stats, trends, searchable, summary) | HERMES `/violations`, `/reports/daily`, `dashboard/` ✅ | Trim to standalone analytics view |
| 8 | Evaluation (Acc/Prec/Rec/F1/mAP + efficiency) | vehicle detector: results.csv, PR, confusion ✅ | **Build:** add rider-model metrics + FPS/latency/size |

**Net new build:** rider-attribute model (helmet/seatbelt), stop-line + illegal-parking rules, annotated-evidence writer, SQLite store, standalone demo app, rider-model evaluation, submission assets.

---

## 3. Phased plan — training is the long pole

The training job runs for hours. We **start it first** (Phase 0→1), then build Phases 2–6 in parallel while the GPU works. Nothing in Phases 2–6 is blocked by training except final integration of the helmet/seatbelt weights (we build against a stub and swap the weights in at the end).

### Phase 0 — Data prep (kick off, then move on)
- Confirm theme = Problem 2.
- Source helmet/seatbelt/rider datasets (Indian where possible). Candidate public sets: Roboflow "Helmet Detection / Rider with/without helmet", Kaggle motorcycle-helmet sets, and any in-repo IDD/FGVD frames with two-wheelers for fine-tuning. *(You confirm which dataset you'll train on — or I source links.)*
- Convert to YOLO format. New classes: `helmet`, `no_helmet`, `seatbelt`, `no_seatbelt`, `rider` (rider count → triple-riding later, optional).
- Reuse `datasets/prepare.py` conventions.

### Phase 1 — START TRAINING (long pole, background)
- Launch GPU training of the rider-attribute model (reuse `aria/train.py` / `run_pipeline.py`).
- Log metrics to results.csv as the vehicle model does.
- **While this runs, we do everything below.**

### Phase 2 — Violation engine extensions (CPU, parallel)
- **Stop-line:** vehicle bbox crosses `stop_line_y` while phase = red (extends existing red-light logic).
- **Illegal parking:** vehicle stationary > N sec inside a restricted polygon (uses tracker; for single images, use a "no-parking zone overlap + vehicle present" rule).
- **Helmet/seatbelt:** integration code in `violation.py` reading rider-model output — built now against a stub, real weights swapped in Phase 1 completion.
- Each violation returns a confidence score (task 4).

### Phase 3 — Standalone image demo app (CPU, parallel) ← the reviewer-facing deliverable
- Gradio or Streamlit: **upload image → preprocess → detect → classify violations → OCR plate → annotated image + violation table (type, confidence, plate, timestamp) → download evidence.**
- No Docker / DB / MQTT required. Single `pip install -r` + one command.
- Source of: **Demo Link, Instructions to Run, Screenshots.**

### Phase 4 — Evidence + analytics (CPU, parallel)
- Annotated-evidence image writer (boxes, labels, plate, timestamp).
- SQLite metadata store (violation records, searchable).
- Analytics view: counts by type, trend chart, summary report, search.

### Phase 5 — Evaluation (after each model finishes)
- `model.val()` → per-class Accuracy/Precision/Recall/F1/mAP50 + mAP50-95 for vehicle detector AND rider model.
- Confusion matrices, PR curves.
- Efficiency: FPS, latency per image, model size (MB), on CPU and GPU.
- Export all figures for deck + report.

### Phase 6 — Submission package
- **Title + Description** (paste-ready for the form).
- **Theme:** Automated Photo Identification…
- **Pitch deck** (.pptx) — problem, approach, architecture, results, demo, impact.
- **Repo:** clean README reframed as image violation system; push to GitHub (Repository URL).
- **Source code zip** (≤50MB — exclude model weights/datasets, link them separately).
- **Demo Link** (hosted Gradio/HF Space or localhost instructions).
- **Instructions to Run.**
- **Screenshots** from the demo app.
- **Video:** script/storyboard for you to record (≤ required length).

---

## 4. Submission form → who produces what

| Form field | Source | Owner |
|------------|--------|-------|
| Title* | Phase 6 draft | Claude drafts |
| Description* | Phase 6 draft | Claude drafts |
| Theme* | "Automated Photo Identification…" | You select |
| Snapshots | Phase 3 demo screenshots | Claude generates / you capture |
| Video URL* | Phase 6 script → you record | You |
| Presentation* | Phase 6 .pptx | Claude builds |
| Demo Link* | Phase 3 app (HF Space / localhost) | Claude builds, you host |
| Repository URL* | Cleaned repo | You push |
| Source Code* | Phase 6 zip | Claude builds |
| Instructions to Run* | Phase 6 | Claude drafts |
| Custom Attachment | optional (metrics report PDF) | Claude (optional) |

---

## 5. Risks & mitigations

- **Helmet/seatbelt dataset quality** → start with a known-good Roboflow set; seatbelt is hardest (in-cabin) so scope it as "best-effort, demo a few clear cases."
- **Training time vs deadline** → start NOW; demo app works with vehicle+geometry violations even if rider model lands late; swap weights when ready.
- **Source zip > 50MB** → exclude `.pt` weights + datasets; provide download links + auto-download script.
- **Reviewer can't run it** → standalone app, no Docker; also host a public demo link as backup.

---

## 6. Immediate next actions

1. You confirm the helmet/seatbelt **dataset** (or ask me to source links) and **start Phase 1 training**.
2. In parallel I build **Phase 2 (violation engine)** and **Phase 3 (demo app)** against a stub.
3. As training finishes, we run **Phase 5 evaluation** and swap real weights in.
4. I assemble **Phase 6** (deck, description, run steps, zip) for you to submit.
