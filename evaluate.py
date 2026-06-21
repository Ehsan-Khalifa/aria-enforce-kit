"""
ARIA-Enforce — Model evaluation report (Phase 5).

Computes the required metrics for a trained model on its test split:
Accuracy proxy, Precision, Recall, F1, mAP50, mAP50-95 (overall + per class),
plus computational efficiency (latency, FPS, model size).

Usage:
    python evaluate.py --weights models/helmet/weights/best.pt \
                       --data datasets/helmet_rider --names "With Helmet,Without Helmet,licence"
Outputs a printed table + saves eval_report.json and the ultralytics
confusion-matrix / PR-curve PNGs under models/runs/eval/.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path


def build_data_yaml(data_dir: Path, names: list[str]) -> Path:
    p = data_dir.parent / "data.eval.yaml"
    p.write_text(
        f"path: {data_dir.as_posix()}\ntrain: train/images\nval: valid/images\n"
        f"test: test/images\nnc: {len(names)}\nnames: {names}\n", encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="dataset dir (has train/valid/test)")
    ap.add_argument("--names", required=True, help="comma-separated class names")
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    names = [n.strip() for n in args.names.split(",")]
    data_yaml = build_data_yaml(Path(args.data), names)
    model = YOLO(args.weights)

    m = model.val(data=str(data_yaml), split="test", imgsz=args.imgsz,
                  device=args.device, project="models/runs", name="eval",
                  exist_ok=True)

    P = float(m.box.mp); R = float(m.box.mr)
    f1 = 2 * P * R / (P + R + 1e-9)
    report = {
        "weights": args.weights,
        "overall": {"precision": round(P, 4), "recall": round(R, 4),
                    "f1": round(f1, 4), "mAP50": round(float(m.box.map50), 4),
                    "mAP50_95": round(float(m.box.map), 4)},
        "per_class": {},
    }
    for i, c in enumerate(names):
        try:
            p_i, r_i, ap50_i, ap_i = m.box.class_result(i)
            f1_i = 2 * p_i * r_i / (p_i + r_i + 1e-9)
            report["per_class"][c] = {"precision": round(float(p_i), 4),
                                      "recall": round(float(r_i), 4),
                                      "f1": round(float(f1_i), 4),
                                      "mAP50": round(float(ap50_i), 4),
                                      "mAP50_95": round(float(ap_i), 4)}
        except Exception:
            pass

    # ── efficiency ──
    import numpy as np
    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)  # warmup
    t0 = time.time(); N = 30
    for _ in range(N):
        model.predict(dummy, verbose=False)
    lat = (time.time() - t0) / N
    size_mb = Path(args.weights).stat().st_size / 1e6
    report["efficiency"] = {"latency_ms": round(lat * 1000, 1),
                            "fps": round(1 / lat, 1),
                            "model_size_mb": round(size_mb, 1)}

    Path("eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nSaved eval_report.json + curves under models/runs/eval/")


if __name__ == "__main__":
    main()
