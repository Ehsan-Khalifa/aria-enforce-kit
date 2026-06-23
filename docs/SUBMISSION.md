# Round 2 Submission — paste-ready content

Copy each block into the matching field on the submission form.

---

## Title
**ARIA-Enforce — AI Traffic Violation Detection & Automated e-Challan for Indian Roads**

---

## Theme
*Automated Photo Identification and Classification for Traffic Violations Using Computer Vision*

---

## Description

**ARIA-Enforce turns a single traffic photo into court-ready enforcement.** Upload an image and the system detects road users, identifies violations, reads the number plate, scores severity, and auto-generates an e-challan — in seconds, with no manual review.

It is purpose-built for **Indian traffic**: a YOLO detector trained on **11 Indian vehicle/road-user classes** (two-wheelers, auto-rickshaws, e-rickshaws, tempos, animal carts, pedestrians and more), plus a dedicated helmet/plate model. It is robust to real-world conditions — low light, rain, fog and glare are handled by a CLAHE + dark-channel preprocessing stage before detection.

**Violations detected (the full required set):**
helmet non-compliance, seatbelt non-compliance, triple-riding, wrong-side driving, stop-line violation, red-light violation, and illegal parking — plus pedestrian-crossing blocking.

**The 8 required capabilities, end to end:**
1. *Preprocessing* — low-light/rain/fog/blur enhancement
2. *Detection* — vehicles, riders, pedestrians, with class labels
3. *Violation detection* — the seven violation types above
4. *Classification + confidence* — every violation typed and scored
5. *License-plate recognition* — OCR with privacy-preserving anonymization
6. *Evidence generation* — annotated images + a searchable metadata store
7. *Analytics & reporting* — counts, trends, daily summaries, search
8. *Evaluation* — Precision, Recall, F1, mAP per class + latency/FPS/size

**Beyond the brief.** ARIA-Enforce doesn't stop at detection — it *acts*:
- **Automated e-challan** generation with the relevant Motor Vehicles Act section and fine.
- **Severity / risk ranking** so enforcement targets the most dangerous offences first, with **repeat-offender escalation**.
- **Event & Congestion Alerts** — a predictive layer mined from 8,000+ historical traffic events that flags congestion hotspots *before* they peak.

**Part of a bigger platform.** The violation engine is one agent (ARIA) of a complete **Intelligent Traffic Management System (ITMS)**: a secure backend (HERMES), a reinforcement-learning adaptive signal controller (NEXUS), and an embedded hardware controller (ATLAS). The same backend that flags a red-light runner can re-time the signal — enforcement and flow optimisation in one system.

**Privacy by design.** Raw plate text never leaves the OCR module; only anonymized, hashed identifiers are stored.

**Results.** Helmet model (YOLO26n), trained on a per-rider **detection** dataset so it judges each rider independently in real multi-rider scenes: **mAP50 0.89, F1 0.85** on a held-out test set (With Helmet 0.92 · Without Helmet 0.87 mAP50), real-time in a 5.3 MB model. The vehicle detector adds 11 Indian classes at mAP50 0.65. The demo runs on a single command (or a double-click `run_demo.bat`) — no Docker, no database server.

---

## Instructions to Run

**Requirements:** Python 3.10/3.11, an NVIDIA GPU recommended for training.

```bash
# 1. Clone + environment
git clone <repository-url> aria-enforce-kit
cd aria-enforce-kit
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install (RTX 50-series first install CUDA 12.8 torch):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 3. Dataset — download Roboflow "Helmet and No Helmet Rider Detection v5"
#    (YOLOv8 export) into datasets/helmet_rider/ (train/valid/test)

# 4. Train the helmet/plate model (~20–40 min on an RTX 5060)
python train_helmet.py --device 0

# 5. Run the demo (opens a local web UI)
python -m aria_enforce.app
#    Tab 1: upload a traffic photo -> violations, plate, e-challans, evidence
#    Tab 2: analytics & reports     Tab 3: event/congestion alerts

# 6. Evaluate
python evaluate.py --weights models/helmet/weights/best.pt \
    --data datasets/helmet_rider --names "With Helmet,Without Helmet,licence"
```
Full step-by-step details, dataset link and troubleshooting are in `README.md`.

---

## Repository URL
`https://github.com/<your-username>/aria-enforce-kit`  *(push the `aria-enforce-kit/` folder)*

## Demo Link
Run locally per the instructions above (`python -m aria_enforce.app`), or host the Gradio app on a free Hugging Face Space and paste that URL here.

## Source Code
Upload `aria-enforce-kit/` as a zip (the `.gitignore` keeps datasets/models out, so it's small).

## Snapshots / Video
Capture screenshots from the three demo tabs (detection result with annotated evidence + e-challan, analytics dashboard, congestion alerts) for the Snapshots field, and record a 2–3 min screen walkthrough of the same flow for the Video URL.
