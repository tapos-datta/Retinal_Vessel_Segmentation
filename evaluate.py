"""Evaluate a trained run on its test images (and optionally on an external, never-seen dataset).

Metrics are computed inside the FOV at the ORIGINAL resolution, with 8-fold TTA by default.

Usage:
    python evaluate.py runs/hybrid/lodo/lodo_HRF/seed1
    python evaluate.py runs/hybrid/all/all/seed1 --external data/external_FIVES      # -> <run>/external_FIVES/
    python evaluate.py runs/hybrid/lodo/lodo_HRF/seed1 --no-tta --save-probs
"""
import argparse
import csv
import time
from pathlib import Path

from u2nete.utils import ROOT, configure_backend, get_device, load_json, save_json

import numpy as np
import torch

from u2nete.data import iter_samples, load_metadata
from u2nete.infer import predict_sample
from u2nete.io import read_binary, write_png
from u2nete.metrics import METRIC_KEYS, compute
from u2nete.model import build_model


def summarize(rows):
    out = {"n": len(rows), "overall": {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}}
    out["overall_sd"] = {k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0 for k in METRIC_KEYS}
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    out["by_dataset"] = {d: {"n": len(v), **{k: float(np.mean([r[k] for r in v])) for k in METRIC_KEYS}}
                         for d, v in sorted(by_ds.items())}
    return out


def evaluate_run(run_dir, tta=True, external=None, name="test", save_probs=False, device_name="auto"):
    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config.json")
    device = get_device(device_name)
    configure_backend(device)
    ck = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    if external:
        pool = Path(external)
        pool = pool if pool.is_absolute() else ROOT / pool
        ids = sorted(load_metadata(pool))
    else:
        pool = Path(cfg["data"]["pool"])
        pool = pool if pool.is_absolute() else ROOT / pool
        ids = load_json(run_dir / "fold.json")["test"]
    meta = load_metadata(pool)

    out_dir = run_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for s in iter_samples(ids, pool, cfg["data"], meta):
        if int(float(meta[s.id].get("vessel_pixels", 1) or 0)) == 0:
            print(f"  {s.dataset:9} {s.id:22} blank reference label: skipped (cannot be scored)")
            continue
        prob = predict_sample(model, s, cfg, device, tta=tta, to_original=True)
        gt = read_binary(pool / "masks" / f"{s.id}.png")
        fov = read_binary(pool / "fov" / f"{s.id}.png")
        m = compute(prob, gt, fov, cfg["inference"]["threshold"])
        rows.append({"id": s.id, "dataset": s.dataset, "condition": meta[s.id].get("condition", ""), **m})
        if save_probs:
            write_png(out_dir / "probs" / f"{s.id}.png", (prob * 255).round().astype(np.uint8))
        print(f"  {s.dataset:9} {s.id:22} Dice {m['dice']:.4f}  clDice {m['cldice']:.4f}  AUC {m['auc']:.4f}  "
              f"Prec {m['precision']:.4f}  Sens {m['sensitivity']:.4f}")

    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dataset", "condition", *METRIC_KEYS])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    summary = {**summarize(rows), "tta": tta, "checkpoint_iteration": ck["iteration"],
               "val_dice": ck["val_dice"], "minutes": round((time.time() - t0) / 60, 1)}
    save_json(summary, out_dir / "summary.json")
    o = summary["overall"]
    print(f"[{run_dir}] {name}: Dice {o['dice']:.4f} ± {summary['overall_sd']['dice']:.4f}  clDice {o['cldice']:.4f}  "
          f"AUC {o['auc']:.4f}  Prec {o['precision']:.4f}  Sens {o['sensitivity']:.4f}  (n={len(rows)})")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--external", help="a pool folder made by prepare_data.py for an unseen dataset")
    ap.add_argument("--name", default=None, help="output sub-folder (default: test, or external_<folder>)")
    ap.add_argument("--save-probs", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    name = args.name or (f"external_{Path(args.external).name}" if args.external else "test")
    evaluate_run(args.run_dir, tta=not args.no_tta, external=args.external, name=name,
                 save_probs=args.save_probs, device_name=args.device)


if __name__ == "__main__":
    main()
