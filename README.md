# ARIA-Enforce — AI Traffic Violation Detection (Training + Demo Kit)

Self-contained kit to **run** the image-based traffic-violation demo and
**retrain** the helmet detector. Theme: *Automated Photo Identification &
Classification for Traffic Violations Using Computer Vision.*

Upload a traffic photo → it detects riders, flags violations (helmet, stop-line,
red-light, illegal parking, …), reads the number plate, scores severity, and
issues an **e-challan** with annotated evidence. Includes an analytics dashboard
and a predictive **congestion-forecast** panel.

A trained helmet model (`models/helmet/weights/best.pt`, ~5 MB) ships with the
repo, so the demo works right after install.

---

## Quick start

### Windows (easiest — double-click)
1. Double-click **`setup.bat`** once (creates a venv + installs everything).
2. Double-click **`run_demo.bat`** — a browser tab opens with the demo.

### Any OS (terminal)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m aria_enforce.app            # add --share for a public link
```
Open the printed URL (e.g. http://127.0.0.1:7860).

> **RTX 50-series (Blackwell) GPUs:** install a CUDA 12.8 PyTorch first —
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

The demo tabs:
- **Detect Violations** — upload a photo → annotated evidence + a table of
  violations and auto-generated e-challans.
- **Analytics & Reports** — counts by type, daily revenue, trend, search.
- **Congestion Forecast** — hotspots + pre-emptive risk alerts (needs
  `data/astram_events.csv`).

---

## Retrain the helmet model (optional)

The helmet detector must be trained on a **detection-style** dataset (per-rider
bounding boxes), NOT a whole-image classifier dataset — otherwise it flags every
person. Use the included verifier to be sure.

1. Download a helmet **detection** dataset (YOLOv8 export) with classes like
   `With Helmet` / `Without Helmet` (real traffic scenes, multiple boxed riders).
   Unzip it into `datasets/helmet_det/` so `train/ valid/ test/ data.yaml` sit
   directly inside.
2. **Verify it first:**
   ```bash
   python verify_dataset.py --data datasets/helmet_det
   ```
   Proceed only if it reports **DETECTION-STYLE** (boxes/image > 1.3,
   full-frame < 25%).
3. **Train** (dataset-agnostic — reads the dataset's own classes):
   ```bash
   python train_helmet.py --data datasets/helmet_det --device 0
   ```
   Defaults: YOLO26n, 60 epochs. On success it writes
   `models/helmet/weights/best.pt`, which the app loads automatically.
4. **Evaluate:**
   ```bash
   python evaluate.py --weights models/helmet/weights/best.pt \
       --data datasets/helmet_det --names "With Helmet,Without Helmet"
   ```

**Current model** (YOLO26n on a per-rider detection dataset): mAP50 **0.89**,
F1 0.85 (With Helmet 0.92 / Without Helmet 0.87 mAP50), ~5 MB, real-time.

---

## Notes
- The large 11-class Indian **vehicle detector** and **PaddleOCR** are not
  bundled here (size). Helmet violations + e-challans work without them; the full
  local system (in the main project) adds vehicle-based stop-line/parking + OCR.
- The app **degrades gracefully** — any missing model/library is skipped and
  shown in the status line rather than crashing.

## Layout
```
setup.bat / run_demo.bat / run_demo.sh   one-click setup & launch
train_helmet.py        train on datasets/helmet_det
verify_dataset.py      check a dataset is detection-style before training
evaluate.py            per-class metrics + efficiency report
requirements.txt
models/helmet/weights/best.pt            trained helmet model (ships with repo)
datasets/helmet_det/   put a detection dataset here to retrain (gitignored)
data/                  optional astram_events.csv for the congestion tab
aria_enforce/
  app.py               Gradio demo (3 tabs)
  image_pipeline.py    orchestrator (adaptive helmet stage)
  config.py            violation catalog + fines + severity
  image_violations.py  helmet/triple + geometry rules
  rider_model.py severity.py challan.py store.py annotate.py event_alert.py
  vision/              detector / ocr / weather modules
```
