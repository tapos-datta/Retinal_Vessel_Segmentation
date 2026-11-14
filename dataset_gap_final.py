"""Dataset-gap histograms for the final models: do the datasets look alike to the network after enhancement?

Rebuilds enhancement_histograms.png of visualize_enhancement.py with the final seed-1 models and FIVES added.
  panels   green channel | CLAHE (green) | hybrid | hybrid_weber   (enhancement front-end output, run on its own)
  curves   background-intensity distribution inside the FOV (GT = 0), one per dataset: CHASE_DB1, DRIVE, HRF, STARE
           (each through the seed-1 LODO model that held that dataset out) and FIVES (seed-1 model trained on all 113
           pooled images; the 792 images with a non-blank label, as in evaluate.py)

Sampling and metric are those of visualize_enhancement.py: per image and representation 4000 random FOV pixels (display
range / overall spread) and 4000 random background pixels (histograms), numpy default_rng(0); per image the background
median; per dataset the median of those;
  dataset_gap = mean |median_a - median_b| over dataset pairs / IQR of the pooled FOV-pixel samples.
Two gaps per panel: over the four training datasets (pairs and IQR from those datasets' images only, the existing
definition) and over all five (pairs and IQR from all images).

Outputs (figures/dataset_gap_final/): dataset_gap_final.png, dataset_gap_final_caption.md, summary.json, per_image.csv

Usage:
    python dataset_gap_final.py --device cpu
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

from u2nete.utils import get_device

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from u2nete.data import fov_mask_image, fov_normalize, load_metadata, preprocess, resize_sample, working_scale
from u2nete.io import read_binary, read_rgb
from visualize_enhancement import COLORS, DATASETS, INK, MUTED, contrast_stats, enhance_full, load_variant, style

SEED = 1
VARIANTS = ["hybrid", "hybrid_weber"]
REPS = ["green", "clahe", *VARIANTS]
REP_LABEL = {"green": "Green channel", "clahe": "CLAHE (green)", "hybrid": "hybrid (enhanced)",
             "hybrid_weber": "hybrid_weber (enhanced)"}
ALL5 = [*DATASETS, "FIVES"]
COLORS5 = {**COLORS, "FIVES": "#e87ba4"}  # categorical slot 5; 5-slot palette validated (direct labels on every curve)
GROUPS = [  # (dataset label, pool, split, fold)
    *[(d, "data/pool", "lodo", f"lodo_{d}") for d in DATASETS],
    ("FIVES", "data/external_FIVES", "all", "all"),
]


def gap(rows, samples, datasets, k):
    """dataset_gap exactly as visualize_enhancement.py, restricted to `datasets` (pairs and IQR)."""
    allpix = np.concatenate([s for s, r in zip(samples[k], rows) if r["dataset"] in datasets])
    iqr = float(np.subtract(*np.percentile(allpix, [75, 25])))
    med = {d: float(np.median([r[f"{k}_background_median"] for r in rows if r["dataset"] == d])) for d in datasets}
    vals = list(med.values())
    g = float(np.mean([abs(a - b) for x, a in enumerate(vals) for b in vals[x + 1:]]) / iqr)
    return g, med, iqr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures/dataset_gap_final")
    ap.add_argument("--device", default="auto",
                    help="auto (cuda, then mps, then cpu), or e.g. cpu / cuda:1; use cpu to leave a shared GPU alone")
    args = ap.parse_args()
    device = get_device(args.device)
    print(f"device: {device}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = Path(args.runs)

    rng = np.random.default_rng(0)
    samples = {k: [] for k in REPS}                 # per image, aligned with rows
    bg = {k: {d: [] for d in ALL5} for k in REPS}
    rows, ckpts, skipped = [], {}, []
    for ds, pool, split, fold in GROUPS:
        pool = Path(pool)
        meta = load_metadata(pool)
        ids = sorted(i for i in meta if meta[i]["dataset"] == ds)
        models, cfgs = {}, {}
        for v in VARIANTS:
            path = runs / v / split / fold / f"seed{SEED}" / "best.pt"
            if not path.exists():
                raise SystemExit(f"missing checkpoint {path}")
            models[v], cfgs[v] = load_variant(v, path, device)
        ckpts[ds] = f"{runs}/{{{','.join(VARIANTS)}}}/{split}/{fold}/seed{SEED}/best.pt"
        for n, i in enumerate(ids):
            if int(float(meta[i].get("vessel_pixels", 1) or 0)) == 0:
                skipped.append(i)  # blank reference label: no vessel/background split (evaluate.py skips it too)
                continue
            prep = {v: (working_scale(meta[i], c), bool(c.get("fov_norm", False)), bool(c.get("mask_fov", False)))
                    for v, c in cfgs.items()}
            if len(set(prep.values())) != 1:
                raise SystemExit(f"{i}: hybrid and hybrid_weber use different input preprocessing {prep}")
            scale, fov_norm, mask_fov = prep[VARIANTS[0]]
            rgb = read_rgb(pool / "images" / f"{i}.png")
            gt = read_binary(pool / "masks" / f"{i}.png")
            fov = read_binary(pool / "fov" / f"{i}.png")
            rgb, gt, fov = resize_sample(rgb, gt, fov, scale)
            model_rgb = fov_normalize(rgb, fov) if fov_norm else (fov_mask_image(rgb, fov) if mask_fov else rgb)
            reps = {"green": rgb[..., 1].astype(np.float32) / 255.0,
                    "clahe": preprocess(rgb, "clahe")[..., 0].astype(np.float32) / 255.0}
            for v in VARIANTS:
                reps[v] = enhance_full(models[v], model_rgb, device, fov=fov)
            row = {"id": i, "dataset": ds}
            for k in REPS:
                img = reps[k]
                row.update({f"{k}_{m}": round(x, 5) for m, x in contrast_stats(img, gt, fov).items()})
                samples[k].append(rng.choice(img[fov > 0], 4000))
                bg[k][ds].append(rng.choice(img[(fov > 0) & (gt == 0)], 4000))
            rows.append(row)
            if (n + 1) % 50 == 0 or n + 1 == len(ids):
                print(f"{ds:9} {n + 1}/{len(ids)}")
        del models

    with open(out / "per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"seed": SEED, "checkpoints": ckpts, "skipped_blank_labels": skipped,
               "n_images": {d: sum(r["dataset"] == d for r in rows) for d in ALL5}, "representations": {}}
    for k in REPS:
        g4, med4, iqr4 = gap(rows, samples, DATASETS, k)
        g5, med5, iqr5 = gap(rows, samples, ALL5, k)
        summary["representations"][k] = {"dataset_gap_4": round(g4, 3), "dataset_gap_5": round(g5, 3),
                                         "iqr_4": iqr4, "iqr_5": iqr5, "background_median": med5}
        print(f"{REP_LABEL[k]:24} dataset gap  4 datasets {g4:.3f}   5 datasets {g5:.3f}   "
              f"(FIVES background median {med5['FIVES']:.3f}; training datasets "
              + ", ".join(f"{d} {med4[d]:.3f}" for d in DATASETS) + ")")
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # --- figure: same construction as enhancement_histograms.png, plus FIVES
    fig, axes = plt.subplots(1, len(REPS), figsize=(4.6 * len(REPS), 4.4), facecolor="white")
    for ax, k in zip(axes, REPS):
        lo, hi = np.percentile(np.concatenate(samples[k]), [1, 99.5])
        if k in VARIANTS:
            bins = np.linspace(lo, hi, 70)
        else:  # 8-bit inputs: align bin edges to the 1/255 grid, otherwise the curves look comb-like
            step = 3 / 255
            bins = np.arange(np.floor(lo * 255) / 255 - 0.5 / 255, hi + step, step)
        for d in ALL5:
            h, e = np.histogram(np.concatenate(bg[k][d]), bins=bins, density=True)
            xs = (e[:-1] + e[1:]) / 2
            ax.plot(xs, h, color=COLORS5[d], linewidth=2, label=d,
                    linestyle=(0, (5, 2)) if d == "FIVES" else "-")
            ax.annotate(d, (xs[np.argmax(h)], h.max()), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
        style(ax)
        ax.set_yticks([])
        s = summary["representations"][k]
        ax.set_title(f"{REP_LABEL[k]}\ngap {s['dataset_gap_4']:.2f} (4 training sets) · "
                     f"{s['dataset_gap_5']:.2f} (with FIVES)", fontsize=10.5, color=INK)
        ax.set_xlabel("background intensity inside the FOV" + (" (own scale)" if k in VARIANTS else ""),
                      fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.suptitle("Do the datasets look alike to the network?  Lower dataset gap = background distributions closer "
                 "together (mean distance between dataset medians ÷ overall spread)", fontsize=10.5, color=MUTED)

    n = summary["n_images"]
    caption = "\n".join([
        "Background-intensity distribution inside the FOV (reference label = 0) per dataset, for the green channel, "
        "CLAHE on the green channel, and the output of the hybrid and hybrid_weber enhancement front-ends run on their "
        "own over the whole image at original resolution. Images: " + ", ".join(f"{d} {n[d]}" for d in ALL5)
        + f" (FIVES: the images with a non-blank label; {len(skipped)} blank-label images left out). FIVES is dashed.",
        "Checkpoints (seed 1; each dataset's images go through the model that never saw that dataset in training, "
        "FIVES through the model trained on all 113 pooled images): "
        + "; ".join(f"{d} {p}" for d, p in ckpts.items()) + ". Green and CLAHE do not depend on a model.",
        "Each panel has its own x-axis. The green and CLAHE panels are 8-bit intensities in [0, 1]; the hybrid and "
        "hybrid_weber panels are in each module's own output units (its own grey range), so positions cannot be "
        "compared across panels, only the spread between curves within a panel.",
        "Dataset gap (as in visualize_enhancement.py): per image the background median, per dataset the median of "
        "those; gap = mean absolute difference between dataset medians over all pairs, divided by the IQR of 4000 "
        "random FOV pixels per image. '4 training sets' uses CHASE_DB1, DRIVE, HRF and STARE only (pairs and IQR), the "
        "existing definition; 'with FIVES' adds FIVES to the pairs and to the IQR. With 4000 pixels per image, the "
        f"FIVES images ({n['FIVES']} of {sum(n.values())}) dominate the IQR of the five-dataset gap.",
    ])
    wrapped = "\n".join(textwrap.fill(p, 205) for p in caption.split("\n"))
    n_lines = wrapped.count("\n") + 1
    bottom = (0.16 * n_lines + 0.15) / fig.get_figheight()
    fig.set_figheight(fig.get_figheight() + 0.16 * n_lines + 0.15)
    bottom = (0.16 * n_lines + 0.15) / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.text(0.01, bottom - 0.01, wrapped, ha="left", va="top", fontsize=8.5, color=INK)
    fig.savefig(out / "dataset_gap_final.png", dpi=130)
    plt.close(fig)
    (out / "dataset_gap_final_caption.md").write_text(caption + "\n")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
