"""How much of the Dice error sits in a thin band along the FOV boundary? Diagnostic only: no model is changed.

For the seed-1 LODO test sets (CHASE_DB1, DRIVE, HRF, STARE held out in turn) and the variants rgb, hybrid and
hybrid_weber, every test image is predicted exactly as evaluate.py does (predict_sample, 8-fold TTA, the run's
inference settings, threshold 0.5, original resolution, FOV only). Each image's Dice is checked against the run's
per_image.csv.

Band: FOV pixels whose distance to the FOV boundary is <= f * fov_diameter (pool metadata), f in BAND_FRACS
(main f = 0.02). The boundary includes the image frame where the FOV mask is cut off by it (HRF), because scoring stops
there too; the alternative without the frame is reported as well ("noframe").

Dice is not additive over regions, so the hybrid_weber - hybrid difference is split exactly with a two-player
Shapley decomposition. D(b, i) = Dice of the prediction that takes the band from model b and the interior from model i:
  band     = 1/2 [ D(W,W) - D(H,W) + D(W,H) - D(H,H) ]
  interior = 1/2 [ D(W,W) - D(W,H) + D(H,W) - D(H,H) ]          band + interior = D(W,W) - D(H,H)
Per-image values are averaged over the images of a dataset (the same per-image mean as the reported Dice).
The same split is reported for hybrid_weber - rgb and hybrid - rgb.

Probability maps are cached as uint8 PNG in results/fov_edge/probs/<variant>/<fold>/ (round(p*255) >= 128 exactly
when p >= 0.5, so thresholded predictions are unchanged); a rerun reuses them.

Outputs (results/fov_edge/): per_image.csv, summary.json, summary.md

Usage:
    python fov_edge.py --device cpu
"""
import argparse
import csv
import json
from pathlib import Path

from u2nete.utils import ROOT, configure_backend, get_device, load_json

import cv2
import numpy as np

from prediction_figure import per_image_dice
from u2nete.data import iter_samples, load_metadata
from u2nete.infer import predict_sample
from u2nete.io import read_binary, write_png
from visualize_enhancement import load_variant

SEED = 1
VARIANTS = ["rgb", "hybrid", "hybrid_weber"]
DATASETS = ["CHASE_DB1", "DRIVE", "HRF", "STARE"]
POOL = ROOT / "data/pool"
THRESHOLD = 0.5
BAND_FRACS = [0.01, 0.02, 0.04]
MAIN = 0.02
PAIRS = [("hybrid_weber", "hybrid"), ("hybrid_weber", "rgb"), ("hybrid", "rgb")]


def dice(pred, gt):
    tp = np.count_nonzero(pred & gt)
    return 2 * tp / max(2 * tp + np.count_nonzero(pred & ~gt) + np.count_nonzero(~pred & gt), 1)


def boundary_distance(fov, frame=True):
    """Distance (px) of each FOV pixel to the nearest non-FOV pixel; with frame=True the image frame counts too."""
    f = (fov > 0).astype(np.uint8)
    if frame:
        f = np.pad(f, 1)
    d = cv2.distanceTransform(f, cv2.DIST_L2, 5)
    return d[1:-1, 1:-1] if frame else d


def prob_for(model, cfg, img_id, meta, device, cache):
    if cache.exists():
        return cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    sample = next(iter_samples([img_id], POOL, cfg["data"], meta))
    prob = predict_sample(model, sample, cfg, device, tta=True, to_original=True)
    q = (prob * 255).round().astype(np.uint8)
    write_png(cache, q)
    return q.astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/fov_edge")
    ap.add_argument("--device", default="auto",
                    help="auto (cuda, then mps, then cpu), or e.g. cpu / cuda:1; use cpu to leave a shared GPU alone")
    args = ap.parse_args()
    device = get_device(args.device)
    configure_backend(device)
    print(f"device: {device}")
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(POOL)

    rows, mismatches = [], []
    for ds in DATASETS:
        fold = f"lodo_{ds}"
        rds = {v: runs / v / "lodo" / fold / f"seed{SEED}" for v in VARIANTS}
        refs = {v: per_image_dice(rds[v] / "test" / "per_image.csv") for v in VARIANTS}
        ids = sorted(set.intersection(*(set(r) for r in refs.values())))
        preds = {v: {} for v in VARIANTS}
        for v in VARIANTS:
            cfg = load_json(rds[v] / "config.json")
            assert cfg["inference"]["threshold"] == THRESHOLD
            model = None
            for i in ids:
                cache = out / "probs" / v / fold / f"{i}.png"
                if model is None and not cache.exists():
                    model, _ = load_variant(v, rds[v] / "best.pt", device)
                preds[v][i] = prob_for(model, cfg, i, meta, device, cache) >= THRESHOLD
            del model
            print(f"{ds:9} {v:13} {len(ids)} images predicted")

        for i in ids:
            gt = read_binary(POOL / "masks" / f"{i}.png") > 0
            fov = read_binary(POOL / "fov" / f"{i}.png") > 0
            diam = float(meta[i]["fov_diameter"])
            dist = {"frame": boundary_distance(fov, True), "noframe": boundary_distance(fov, False)}
            g = gt[fov]
            p = {v: preds[v][i][fov] for v in VARIANTS}
            row = {"dataset": ds, "id": i}
            for v in VARIANTS:
                d = dice(p[v], g)
                row[f"dice_{v}"] = d
                if abs(d - refs[v][i]) > 5e-4:
                    mismatches.append(f"{ds} {i} {v}: {d:.4f} vs evaluate.py {refs[v][i]:.4f}")
            for kind, dd in dist.items():
                dvals = dd[fov]
                for frac in BAND_FRACS:
                    band = dvals <= frac * diam
                    tag = f"{kind}_{frac:g}"
                    row[f"band_area_{tag}"] = float(band.mean())
                    row[f"gt_in_band_{tag}"] = float((g & band).sum() / max(g.sum(), 1))
                    for v in VARIANTS:
                        fp, fn = p[v] & ~g, ~p[v] & g
                        row[f"fp_band_{v}_{tag}"] = int((fp & band).sum())
                        row[f"fp_int_{v}_{tag}"] = int((fp & ~band).sum())
                        row[f"fn_band_{v}_{tag}"] = int((fn & band).sum())
                        row[f"fn_int_{v}_{tag}"] = int((fn & ~band).sum())
                    for a, b in PAIRS:
                        D = lambda bm, im: dice(np.where(band, p[bm], p[im]), g)
                        aa, bb, ab, ba = D(a, a), D(b, b), D(a, b), D(b, a)  # ab: band from a, interior from b
                        row[f"band_{a}-{b}_{tag}"] = 0.5 * ((aa - ba) + (ab - bb))
                        row[f"int_{a}-{b}_{tag}"] = 0.5 * ((aa - ab) + (ba - bb))
            rows.append(row)

    with open(out / "per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    # --- summary per dataset and over all four LODO test sets (mean of per-image values)
    summary = {"seed": SEED, "band_fracs": BAND_FRACS, "main": MAIN, "datasets": {}}
    groups = {**{ds: [r for r in rows if r["dataset"] == ds] for ds in DATASETS}, "all (113)": rows}
    for name, rs in groups.items():
        s = {"n": len(rs), **{f"dice_{v}": float(np.mean([r[f"dice_{v}"] for r in rs])) for v in VARIANTS}}
        for kind in ("frame", "noframe"):
            for frac in BAND_FRACS:
                tag = f"{kind}_{frac:g}"
                s[tag] = {"band_area": float(np.mean([r[f"band_area_{tag}"] for r in rs])),
                          "gt_in_band": float(np.mean([r[f"gt_in_band_{tag}"] for r in rs]))}
                for v in VARIANTS:
                    e_band = sum(r[f"fp_band_{v}_{tag}"] + r[f"fn_band_{v}_{tag}"] for r in rs)
                    e_all = e_band + sum(r[f"fp_int_{v}_{tag}"] + r[f"fn_int_{v}_{tag}"] for r in rs)
                    s[tag][f"error_share_in_band_{v}"] = e_band / max(e_all, 1)
                    s[tag][f"fp_band_{v}"] = sum(r[f"fp_band_{v}_{tag}"] for r in rs)
                    s[tag][f"fn_band_{v}"] = sum(r[f"fn_band_{v}_{tag}"] for r in rs)
                for a, b in PAIRS:
                    band = float(np.mean([r[f"band_{a}-{b}_{tag}"] for r in rs]))
                    inter = float(np.mean([r[f"int_{a}-{b}_{tag}"] for r in rs]))
                    s[tag][f"{a}-{b}"] = {"total": band + inter, "band": band, "interior": inter}
        summary["datasets"][name] = s
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # --- markdown table: main band, both boundary definitions, all band widths for hybrid_weber - hybrid
    md = ["# FOV-edge diagnostic (seed 1, LODO test sets, 8-fold TTA, threshold 0.5)", "",
          f"Band = FOV pixels within f x FOV diameter of the FOV boundary (image frame counts as boundary unless "
          f"'noframe'). Main f = {MAIN}. Dice differences split exactly into band + interior (two-player Shapley).", ""]
    for kind in ("frame", "noframe"):
        tag = f"{kind}_{MAIN:g}"
        md += [f"## hybrid_weber - hybrid, band f = {MAIN} ({kind})", "",
               "| dataset | n | band area | Dice hybrid | Dice hybrid_weber | total diff | from band | from interior | "
               "band share of diff | error share in band (hybrid / hybrid_weber) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for name, s in summary["datasets"].items():
            t = s[tag]
            d = t["hybrid_weber-hybrid"]
            share = d["band"] / d["total"] if abs(d["total"]) > 1e-9 else float("nan")
            md.append(f"| {name} | {s['n']} | {t['band_area']:.1%} | {s['dice_hybrid']:.4f} | "
                      f"{s['dice_hybrid_weber']:.4f} | {d['total']:+.4f} | {d['band']:+.4f} | {d['interior']:+.4f} | "
                      f"{share:.0%} | {t['error_share_in_band_hybrid']:.1%} / {t['error_share_in_band_hybrid_weber']:.1%} |")
        md.append("")
    md += ["## Band contribution vs band width (frame), all pairs", "",
           "| dataset | pair | " + " | ".join(f"band f={f:g}" for f in BAND_FRACS) + " | total |",
           "|---|---|" + "---|" * (len(BAND_FRACS) + 1)]
    for name, s in summary["datasets"].items():
        for a, b in PAIRS:
            md.append(f"| {name} | {a} - {b} | " + " | ".join(
                f"{s[f'frame_{f:g}'][f'{a}-{b}']['band']:+.4f}" for f in BAND_FRACS)
                + f" | {s[f'frame_{MAIN:g}'][f'{a}-{b}']['total']:+.4f} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if mismatches:
        print("WARNING: Dice differs from evaluate.py per_image.csv:\n  " + "\n  ".join(mismatches))
    else:
        print(f"all {len(rows) * len(VARIANTS)} Dice values match evaluate.py per_image.csv")


if __name__ == "__main__":
    main()
