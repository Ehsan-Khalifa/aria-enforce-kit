# ARIA-Enforce — AI Traffic Violation Detection (Training + Demo Kit)

Self-contained kit to **train** the helmet/plate model and **run** the
image-based traffic-violation demo. Theme: *Automated Photo Identification &
Classification for Traffic Violations Using Computer Vision.*

Detects: helmet non-compliance, seatbelt, triple-riding, stop-line, red-light,
illegal parking, wrong-side, pedestrian-crossing — then issues an **e-challan**
with severity scoring and stores annotated evidence. Includes an analytics
dashboard and a predictive **event/congestion** panel.

---

## 0. Prerequisites
- Python 3.10 or 3.11
- An NVIDIA GPU strongly recommended for training (CPU works but is slow)
- ~3 GB free disk for the dataset

## 1. Get the code
```bash
git clone <your-repo-url> aria-enforce-kit
cd aria-enforce-kit
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

## 2. Install dependencies

**If you have an RTX 50-series (5060/5070/5080/5090 — "Blackwell"), install
PyTorch FIRST (older torch will fail with sm_120 / "no kernel image" errors):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
Then, for everyone:
```bash
pip install -r requirements.txt
```
Verify the GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `True  NVIDIA GeForce RTX 5060 ...`

## 3. Download the dataset
Download **Helmet and No Helmet Rider Detection v5** from Roboflow
(https://universe.roboflow.com/gw-khadatkar-and-sv-wasule/helmet-and-no-helmet-rider-detection)
→ **Download Dataset → Format: YOLOv8 → zip**, then unzip into
`datasets/helmet_rider/` so it looks like:
```
datasets/helmet_rider/train/images/  + labels/
datasets/helmet_rider/valid/images/  + labels/
datasets/helmet_rider/test/images/   + labels/
```
(More detail in `datasets/helmet_rider/PLACEHOLDER.md`.)

## 4. Train  ← the long step, do this on the GPU machine
```bash
python train_helmet.py --device 0
```
Defaults: **yolo26n, 40 epochs** (fast; ~20–40 min on an RTX 5060-class GPU).
Options:
```bash
python train_helmet.py --device 0 --model m --epochs 60   # higher accuracy, slower
python train_helmet.py --device cpu                        # no GPU (slow)
```
On success it writes `models/helmet/weights/best.pt` and prints test mAP.
**Send `models/helmet/weights/best.pt` back** (it's small) so it can be plugged
into the full system.

## 5. Run the demo
```bash
python -m aria_enforce.app
```
Open the printed URL (e.g. http://127.0.0.1:7860):
- **Tab 1 — Violation Detection:** upload a traffic photo → annotated evidence,
  violations, severity, plate, e-challans.
- **Tab 2 — Analytics & Reports:** counts, trends, daily revenue, search.
- **Tab 3 — Event & Congestion Alerts:** needs `data/astram_events.csv` (optional).

> The app **degrades gracefully**: before training, helmet checks show
> "model pending" but the UI still runs. After step 4, it activates
> automatically. The large vehicle-detection model isn't shipped in this kit
> (too big for GitHub); helmet/seatbelt violations work without it.

## Troubleshooting
- **`CUDA available: False`** → install the cu128 PyTorch (step 2) and update your NVIDIA driver.
- **`Missing datasets/helmet_rider/...`** → finish step 3; folders must be `train/valid/test` directly under `datasets/helmet_rider/`.
- **OCR not working** → `pip install paddleocr paddlepaddle`; optional, plate shows `UNKNOWN` without it.
- **Out of memory while training** → add `--batch 8` (or `--imgsz 480`).

## Layout
```
train_helmet.py            training entry point
requirements.txt
datasets/helmet_rider/     <- put dataset here (gitignored)
data/                      <- optional astram_events.csv
models/                    <- training outputs (gitignored)
aria_enforce/
  app.py                   Gradio demo (3 tabs)
  image_pipeline.py        orchestrator
  config.py                violation catalog + fines + severity
  image_violations.py      helmet/triple + geometry rules
  rider_model.py           helmet model wrapper (stub-capable)
  severity.py challan.py store.py annotate.py event_alert.py
  vision/                  detector / ocr / weather (reused ARIA modules)
```
