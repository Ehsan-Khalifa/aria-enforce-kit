"""
ARIA-Enforce — Dataset sanity check (run BEFORE training).

Flags "classifier-style" datasets (mostly whole-image labels) that look like
detection data but aren't — the trap that made the first helmet model flag every
person. A good DETECTION dataset has several boxes per image and few full-frame
boxes.

Usage:
    python verify_dataset.py --data datasets/helmet_det
"""
from __future__ import annotations
import argparse, glob, os
from collections import Counter


def scan(split_dir):
    labels = glob.glob(os.path.join(split_dir, "labels", "*.txt"))
    n_img = len(labels); n_box = 0; n_full = 0; cls = Counter(); empty = 0
    for f in labels:
        lines = [l.split() for l in open(f).read().split("\n") if l.strip()]
        if not lines:
            empty += 1
        for p in lines:
            if len(p) < 5:
                continue
            n_box += 1; cls[int(float(p[0]))] += 1
            w, h = float(p[3]), float(p[4])
            if w >= 0.95 and h >= 0.95:
                n_full += 1
    return n_img, n_box, n_full, empty, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    tot_img = tot_box = tot_full = tot_empty = 0
    cls = Counter()
    for split in ("train", "valid", "val", "test"):
        d = os.path.join(args.data, split)
        if os.path.isdir(os.path.join(d, "labels")):
            i, b, fl, e, c = scan(d)
            tot_img += i; tot_box += b; tot_full += fl; tot_empty += e; cls += c
            print(f"  {split:6s}: {i:5d} imgs, {b:6d} boxes, "
                  f"{fl} full-frame, {e} empty")
    if tot_img == 0:
        raise SystemExit("No labels found — check the path/structure.")
    bpi = tot_box / tot_img
    pct_full = 100 * tot_full / max(1, tot_box)
    print(f"\n  TOTAL: {tot_img} images, {tot_box} boxes")
    print(f"  boxes/image: {bpi:.2f}")
    print(f"  full-frame boxes: {pct_full:.1f}%")
    print(f"  classes: {dict(cls)}")
    print("\n  VERDICT:", end=" ")
    if pct_full > 25 or bpi < 1.3:
        print("CLASSIFIER-STYLE ❌  — do NOT use for scene detection.\n"
              "  (Mostly whole-image labels / ~1 object per image. Pick a real\n"
              "   detection dataset with multiple boxed riders per image.)")
    else:
        print("DETECTION-STYLE ✅  — good for multi-rider scene training.")


if __name__ == "__main__":
    main()
