"""
ARIA-Enforce — Helmet model training (dataset-agnostic).

Fine-tunes YOLO on any YOLO-format helmet dataset. It reads the dataset's OWN
data.yaml (classes + splits) and only rewrites the `path:` to an absolute path,
so it works wherever the repo is cloned and with whatever classes the dataset
ships (e.g. helmet / no-helmet / rider / number_plate).

IMPORTANT: run `python verify_dataset.py --data <dir>` FIRST. If it reports the
dataset is "classifier-style" (mostly full-frame labels), DO NOT train on it —
pick a real detection dataset. That is exactly the bug that made the first model
flag every person.

Usage (from kit root):
    python verify_dataset.py --data datasets/helmet_det      # check it first
    python train_helmet.py  --data datasets/helmet_det --device 0
    python train_helmet.py  --model m --epochs 60            # higher accuracy
On success, best.pt is copied to models/helmet/weights/best.pt (where the app
loads it).
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "models" / "helmet"


def build_runtime_yaml(data_dir: Path) -> Path:
    """Read the dataset's own data.yaml, keep its classes, fix path to absolute."""
    import yaml
    src = data_dir / "data.yaml"
    if not src.exists():
        raise SystemExit(f"[train] No data.yaml in {data_dir}. Extract a "
                         f"YOLO-format dataset there (with train/valid/test "
                         f"and data.yaml).")
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    names = cfg.get("names")
    nc = cfg.get("nc") or (len(names) if names else 0)
    # locate splits (Roboflow uses train/valid/test)
    def split_dir(*cands):
        for c in cands:
            if (data_dir / c).exists():
                return c
        return cands[0]
    train = split_dir("train/images", "train")
    val = split_dir("valid/images", "val/images", "valid", "val")
    test = split_dir("test/images", "test", val)
    out = ROOT / "data.runtime.yaml"
    out.write_text(
        f"path: {data_dir.as_posix()}\n"
        f"train: {train}\nval: {val}\ntest: {test}\n"
        f"nc: {nc}\nnames: {names}\n", encoding="utf-8")
    print(f"[train] classes ({nc}): {names}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/helmet_det",
                    help="dataset dir containing data.yaml + train/valid/test")
    ap.add_argument("--model", default="n", choices=["n", "s", "m", "l"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu")
    args = ap.parse_args()

    data_dir = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    data_yaml = build_runtime_yaml(data_dir)

    from ultralytics import YOLO
    base = f"yolo26{args.model}.pt"
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
