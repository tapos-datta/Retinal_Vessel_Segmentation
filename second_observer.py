"""Human baseline: how well does a second human annotator agree with the first (the ground truth)?

STARE ships a second labeller (Valentina Kouznetsova, files im0001.vk.ppm), CHASE_DB1 a second observer
(Image_01L_2ndHO.png), DRIVE a second set for its test images (2nd_manual). Download those label files, put
them in one folder, and run:

    python second_observer.py --labels /path/to/second_observer_masks

The same metrics are computed as for the model (inside the FOV, original resolution), so a model Dice close to
the second-observer Dice means the model is near human agreement.
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np

from u2nete.io import IMAGE_EXTS, read_binary
from u2nete.metrics import METRIC_KEYS, compute


def image_id(name: str):
    stem = Path(name).name
    for pat in (r"^(im\d+)\.vk", r"^(Image_\d+[LR])_2ndHO", r"^(\d+)_manual2"):
        m = re.match(pat, stem)
        if m:
            return m.group(1) if not m.group(1).isdigit() else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--out", default="results/second_observer.csv")
    args = ap.parse_args()

    pool = Path(args.pool)
    rows = []
    for p in sorted(Path(args.labels).iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        i = image_id(p.name)
        if not i or not (pool / "masks" / f"{i}.png").exists():
            continue
        gt = read_binary(pool / "masks" / f"{i}.png")
        fov = read_binary(pool / "fov" / f"{i}.png")
        second = read_binary(p).astype(np.float32)
        m = compute(second, gt, fov, threshold=0.5)
        dataset = "STARE" if i.startswith("im") else "CHASE_DB1"
        rows.append({"id": i, "dataset": dataset, **m})
        print(f"{dataset:9} {i:12} Dice {m['dice']:.4f}  clDice {m['cldice']:.4f}  Sens {m['sensitivity']:.4f}")
    if not rows:
        raise SystemExit("no second-observer masks matched images in the pool")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dataset", *METRIC_KEYS])
        w.writeheader()
        w.writerows({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    for d in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == d]
        print(f"\n{d}: second observer vs ground truth, mean Dice {np.mean([r['dice'] for r in sub]):.4f} (n={len(sub)})")


if __name__ == "__main__":
    main()
