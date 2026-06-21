"""
ARIA-Enforce — Helmet + Licence-plate model training.

Fine-tunes YOLO on datasets/helmet_rider (3 classes:
With Helmet / Without Helmet / licence). Robust to wherever the repo is
cloned: it builds an absolute data.yaml at runtime, so paths never break.

Usage (from the kit root):
    python train_helmet.py                      # yolo26n, 40 epochs (recommended)
    python train_helmet.py --model m --epochs 60
    python train_helmet.py --device 0           # pick GPU
    python train_helmet.py --device cpu         # CPU fallback (slow)

On success, best.pt is copied to models/helmet/weights/best.pt — exactly where
the demo app (aria_enforce/app.py) looks for it.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "helmet_rider"
OUT_DIR = ROOT / "models" / "helmet"


def write_data_yaml() -> Path:
    """Generate an absolute-path data.yaml so ultralytics can't get lost."""
    for split in ("train", "valid", "test"):
        if not (DATA_DIR / split / "images").exists():
            raise SystemExit(
                f"[train] Missing {DATA_DIR/split/'images'}.\n"
                f"Download the helmet dataset and extract it into "
                f"{DATA_DIR}\\ so that train/ valid/ test/ sit directly there. "
                f"See datasets/helmet_rider/PLACEHOLDER.md")
    yaml_path = ROOT / "data.runtime.yaml"
    yaml_path.write_text(
        f"path: {DATA_DIR.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: 3\n"
        f"names: ['With Helmet', 'Without Helmet', 'licence']\n",
        encoding="utf-8")
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="n", choices=["n", "s", "m", "l"],
                    help="YOLO size; n (nano) recommended — fast & accurate enough")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    ap.add_argument("--device", default=None, help="e.g. 0, 0,1, cpu")
    args = ap.parse_args()

    data_yaml = write_data_yaml()
    from ultralytics import YOLO

    base = f"yolo26{args.model}.pt"   # ultralytics auto-downloads if absent
    print(f"[train] base={base}  data={data_yaml}  device={args.device}")
    model = YOLO(base)
    results = model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device,
        project=str(ROOT / "models" / "runs"), name="helmet",
        patience=15, fliplr=0.5, mosaic=1.0, close_mosaic=10,
        exist_ok=True, verbose=True)

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        (OUT_DIR / "weights").mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, OUT_DIR / "weights" / "best.pt")
        print(f"[train] OK -> {OUT_DIR/'weights'/'best.pt'}")

    m = model.val(data=str(data_yaml), split="test")
    print(f"[train] test mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}")


if __name__ == "__main__":
    main()
