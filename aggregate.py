"""Collect every finished run into tables and paired statistical tests.

Usage:
    python aggregate.py                       # all runs under runs/, reference variant rgb_scale_strong
    python aggregate.py --ref hybrid --metric cldice
    python aggregate.py --ref rgb --test-dir external_FIVES --by condition --out results/stage3_FIVES   # stage 3

Writes results/summary.md (readable tables) and results/all_per_image.csv (every test prediction).

How numbers are combined:
  * per image: metric averaged over seeds (so each image counts once)
  * per variant: mean over images; "± sd" is the standard deviation across seeds of the per-seed mean
  * LODO: one column per held-out dataset (each column = performance on a dataset never seen in training)
  * paired test: Wilcoxon signed-rank on per-image differences vs the reference variant
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from u2nete.metrics import METRIC_KEYS


def collect(root, test_dir="test"):
    """Images with a blank reference label (no vessels, AUC undefined) cannot be scored and are left out."""
    rows, blank = [], set()
    for f in Path(root).glob(f"*/*/*/seed*/{test_dir}/per_image.csv"):
        variant, split, fold, seed = f.parts[-6], f.parts[-5], f.parts[-4], f.parts[-3]
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["auc"] == "nan":
                    blank.add(r["id"])
                    continue
                rows.append({"variant": variant, "split": split, "fold": fold, "seed": seed,
                             **{k: r[k] for k in ("id", "dataset", "condition")},
                             **{k: float(r[k]) for k in METRIC_KEYS}})
    if blank:
        print(f"left out {len(blank)} images with a blank label: {sorted(blank)}")
    return rows


def fmt(m, s=None):
    return f"{m:.4f}" if s is None or np.isnan(s) else f"{m:.4f} ± {s:.4f}"


def per_seed_mean(rows, metric):
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r[metric])
    means = [np.mean(v) for v in by_seed.values()]
    return float(np.mean(means)), (float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")), len(means)


def image_means(rows, metric):
    acc = defaultdict(list)
    for r in rows:
        acc[r["id"]].append(r[metric])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--ref", default="rgb_scale_strong", help="reference variant for paired tests")
    ap.add_argument("--metric", default="dice", choices=METRIC_KEYS)
    ap.add_argument("--out", default="results")
    ap.add_argument("--test-dir", default="test", help="per-run results folder, e.g. external_FIVES")
    ap.add_argument("--by", choices=["dataset", "condition"], default="dataset",
                    help="table columns: dataset, or condition (useful for FIVES: amd, diabetic_retinopathy, ...)")
    args = ap.parse_args()

    rows = collect(args.runs, args.test_dir)
    if args.by == "condition":
        for r in rows:  # condition becomes the column key; images without one keep their dataset name
            r["dataset"] = r["condition"] or r["dataset"]
    if not rows:
        raise SystemExit(f"no finished runs found under {args.runs}/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "all_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    m = args.metric
    lines = [f"# Results ({m})", "", f"Runs folder: `{args.runs}` · results: `{args.test_dir}` · columns: {args.by} · reference for paired tests: `{args.ref}`", ""]
    for split in sorted({r["split"] for r in rows}):
        srows = [r for r in rows if r["split"] == split]
        variants = sorted({r["variant"] for r in srows})
        datasets = sorted({r["dataset"] for r in srows})
        title = {"lodo": "Leave-one-dataset-out (each column = a dataset never seen in training)",
                 "legacy": "Original 11-image test split",
                 "all": f"Trained on all 113 images, tested on `{args.test_dir}`"}.get(split, f"{split}: cross-validation, every image tested once")
        lines += [f"## {title}", "", "| variant | seeds | " + " | ".join(datasets) + " | **mean** | clDice | AUC | precision | sensitivity |",
                  "|---|---|" + "---|" * (len(datasets) + 5)]
        for v in variants:
            vr = [r for r in srows if r["variant"] == v]
            cells = []
            for d in datasets:
                dr = [r for r in vr if r["dataset"] == d]
                cells.append(fmt(*per_seed_mean(dr, m)[:2]) if dr else "–")
            # LODO mean = mean of the per-dataset means (each unseen domain weighs equally)
            if split == "lodo" and args.by == "dataset":
                overall = np.mean([per_seed_mean([r for r in vr if r["dataset"] == d], m)[0]
                                   for d in datasets if any(r["dataset"] == d for r in vr)])
                ov = fmt(float(overall))
            else:
                ov = fmt(*per_seed_mean(vr, m)[:2])
            extra = [fmt(per_seed_mean(vr, k)[0]) for k in ("cldice", "auc", "precision", "sensitivity")]
            n_seeds = per_seed_mean(vr, m)[2]
            lines.append(f"| {v} | {n_seeds} | " + " | ".join(cells) + f" | **{ov}** | " + " | ".join(extra) + " |")

        ref_rows = [r for r in srows if r["variant"] == args.ref]
        if ref_rows:
            ref = image_means(ref_rows, m)
            lines += ["", f"Paired comparison with `{args.ref}` (per-image {m}, averaged over seeds):", "",
                      "| variant | images | mean difference | images better / worse | Wilcoxon p |", "|---|---|---|---|---|"]
            for v in variants:
                if v == args.ref:
                    continue
                other = image_means([r for r in srows if r["variant"] == v], m)
                common = sorted(set(ref) & set(other))
                if len(common) < 5:
                    continue
                diff = np.array([other[i] - ref[i] for i in common])
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                lines.append(f"| {v} | {len(common)} | {diff.mean():+.4f} | {(diff > 0).sum()} / {(diff < 0).sum()} | {p:.2g} |")
        lines.append("")

    lines += ["Notes: metrics are computed inside the FOV at the original resolution with 8-fold TTA.",
              "With several variants compared, treat p-values above ~0.01 as inconclusive (multiple comparisons)."]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
