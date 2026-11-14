"""Labelling-style figure from results/label_style (existing CSV files; no model is run).

  left   labelled vessel width per dataset, relative to eye size (width_rel = mean labelled width at the centreline /
         FOV diameter, x1000), one dot per image, mean marked; mean width in pixels written beside each dataset
  right  predicted / labelled width (width_ratio) per dataset, one marker per variant (3-seed mean, seed range as a
         bar), LODO models on their held-out dataset, `all` models on FIVES; reference line at 1

Every number shown is checked against results/label_style/summary.md (per-variant LODO and FIVES width_ratio,
the FIVES hybrid - rgb centreline-recall difference, and the label statistics table); any mismatch stops the script.
Averaging follows label_style_test.summarize: per seed, the mean over images (nan ignored) within a dataset, LODO =
mean of the 4 dataset means; then the mean over seeds.

Outputs (figures/report/): label_style_widths.png, label_style_widths_caption.md

Usage:
    python label_style_figure.py
"""
import argparse
import csv
import re
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualize_enhancement import GRID, INK, MUTED, style

SRC = Path("results/label_style")
DATASETS = ["CHASE_DB1", "DRIVE", "HRF", "STARE", "FIVES"]
WANTED = ["rgb", "clahe", "hybrid", "enh3", "hybrid_weber"]
COLORS = {"rgb": "#2a78d6", "clahe": "#eb6834", "hybrid": "#1baf7a", "enh3": "#eda100", "hybrid_weber": "#e87ba4"}


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def per_seed_dataset_means(rows, metric):
    """{(source, variant, seed): {dataset: nanmean over images}}"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[(r["source"], r["variant"], r["seed"])][r["dataset"]].append(fnum(r[metric]))
    return {k: {d: float(np.nanmean(v)) for d, v in by.items()} for k, by in acc.items()}


def summary_value(source, variant, metric, psd):
    """label_style_test.summarize: per seed, mean of dataset means; then mean over seeds."""
    vals = [np.mean(list(by.values())) for (s, v, _), by in psd.items() if s == source and v == variant]
    return float(np.mean(vals))


def parse_summary(text):
    main = {}
    for m in re.finditer(r"^\| (\w+) \| \d+ / \d+ \| (\w+) \| ([-+.\d]+) \| ([-+.\d]+) \|", text, re.M):
        main[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))
    sec = {}
    for m in re.finditer(r"^\| (\w+) − (\w+) \| ([-+.\d]+) \| ([-+.\d]+) \| ([-+.\d]+) \|", text, re.M):
        sec[(m.group(1), m.group(2))] = {"dice": float(m.group(3)), "cl_recall": float(m.group(4)),
                                         "width_ratio": float(m.group(5))}
    lab = {}
    part = text.split("## Label statistics by dataset")[1]
    for m in re.finditer(r"^\| (\w+) \| (\d+) \| ([.\d]+) \| ([.\d]+) \| ([.\d]+) \|", part, re.M):
        lab[m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
    return main, sec, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/report")
    args = ap.parse_args()

    rows = read(SRC / "per_image.csv")
    labels = read(SRC / "label_stats.csv")
    s_main, s_sec, s_lab = parse_summary((SRC / "summary.md").read_text())
    variants = [v for v in WANTED if any(r["variant"] == v for r in rows)]
    absent = [v for v in WANTED if v not in variants]
    psd = {m: per_seed_dataset_means(rows, m) for m in ("width_ratio", "cl_recall")}

    # --- checks against summary.md
    errors = []
    for v in variants:
        for src, j in (("lodo", 0), ("fives", 1)):
            got = round(summary_value(src, v, "width_ratio", psd["width_ratio"]), 4)
            want = s_main.get((v, "width_ratio"), (None, None))[j]
            if want is None or abs(got - want) > 1e-9:
                errors.append(f"width_ratio {v} {src}: computed {got}, summary {want}")
    clr = round(summary_value("fives", "hybrid", "cl_recall", psd["cl_recall"])
                - summary_value("fives", "rgb", "cl_recall", psd["cl_recall"]), 4)
    if abs(clr - s_sec[("hybrid", "rgb")]["cl_recall"]) > 1e-9:
        errors.append(f"hybrid - rgb cl_recall on FIVES: computed {clr}, summary {s_sec[('hybrid', 'rgb')]['cl_recall']}")
    lab = {}
    for d in DATASETS:
        xs = [r for r in labels if r["dataset"] == d]
        lab[d] = {"n": len(xs), "px": [float(r["width_px"]) for r in xs], "rel": [1000 * float(r["width_rel"]) for r in xs]}
        n, px, rel = s_lab[d]
        if (len(xs), round(np.mean(lab[d]["px"]), 2), round(np.mean(lab[d]["rel"]), 2)) != (n, px, rel):
            errors.append(f"label stats {d}: computed {len(xs)}, {np.mean(lab[d]['px']):.2f}, "
                          f"{np.mean(lab[d]['rel']):.2f}; summary {n}, {px}, {rel}")
    if errors:
        raise SystemExit("values differ from results/label_style/summary.md:\n  " + "\n  ".join(errors))
    print("all values match results/label_style/summary.md")

    # per dataset, per variant: 3-seed mean of the dataset mean width_ratio, and the seed range
    wr = {}
    for v in variants:
        for d in DATASETS:
            src = "fives" if d == "FIVES" else "lodo"
            seeds = [by[d] for (s, vv, _), by in psd["width_ratio"].items() if s == src and vv == v and d in by]
            wr[(v, d)] = (float(np.mean(seeds)), min(seeds), max(seeds), len(seeds))
    rel_mean = {d: float(np.mean(lab[d]["rel"])) for d in DATASETS}
    px_mean = {d: float(np.mean(lab[d]["px"])) for d in DATASETS}
    ratio_fh = rel_mean["FIVES"] / rel_mean["HRF"]

    # --- figure
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="white", gridspec_kw={"width_ratios": [1, 1.25]})
    rng = np.random.default_rng(0)
    xs = np.arange(len(DATASETS))
    for x, d in zip(xs, DATASETS):
        vals = np.array(lab[d]["rel"])
        big = d != "FIVES"
        axl.scatter(x + rng.uniform(-0.22, 0.22, len(vals)), vals, s=14 if big else 5, color="#8f8e86",
                    alpha=0.85 if big else 0.3, linewidth=0, zorder=2)
        axl.plot([x - 0.3, x + 0.3], [rel_mean[d]] * 2, color=INK, linewidth=3, solid_capstyle="round", zorder=3)
        axl.annotate(f"{rel_mean[d]:.2f}\n{px_mean[d]:.1f} px", (x, max(vals.max(), rel_mean[d])), xytext=(0, 6),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
    style(axl)
    axl.set_xticks(xs, [f"{d}\n(n={lab[d]['n']})" for d in DATASETS], fontsize=9, color=INK)
    axl.tick_params(axis="x", length=0)
    axl.set_ylim(0, max(max(lab[d]["rel"]) for d in DATASETS) * 1.18)
    axl.set_ylabel("labelled vessel width / FOV diameter (×1000)", fontsize=9.5, color=MUTED)
    axl.set_title(f"Labelled width relative to eye size\nFIVES {ratio_fh:.1f}× HRF; mean in px written above",
                  fontsize=11, color=INK, loc="left")

    k = len(variants)
    off = (np.arange(k) - (k - 1) / 2) * 0.16
    axr.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    axr.text(len(DATASETS) - 0.45, 1.0, "predicted = labelled", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    axr.axvspan(len(DATASETS) - 1.5, len(DATASETS) - 0.5, color="#f1f0ea", zorder=0)
    for j, v in enumerate(variants):
        m = [wr[(v, d)] for d in DATASETS]
        axr.vlines(xs + off[j], [a[1] for a in m], [a[2] for a in m], color=COLORS[v], linewidth=2.5, alpha=0.45, zorder=2)
        axr.scatter(xs + off[j], [a[0] for a in m], s=46, color=COLORS[v], edgecolor="white", linewidth=1.2,
                    zorder=3, label=v)
    style(axr)
    axr.set_xticks(xs, ["CHASE_DB1", "DRIVE", "HRF", "STARE", "FIVES"], fontsize=9, color=INK)
    axr.tick_params(axis="x", length=0)
    axr.set_xlim(-0.5, len(DATASETS) - 0.5)
    axr.set_ylabel("predicted / labelled width (width_ratio)", fontsize=9.5, color=MUTED)
    axr.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left", ncol=len(variants))
    fives_rng = (min(wr[(v, "FIVES")][0] for v in variants), max(wr[(v, "FIVES")][0] for v in variants))
    axr.set_title(f"Predicted vs labelled width: every model draws FIVES vessels thinner\nFIVES {fives_rng[0]:.2f}–"
                  f"{fives_rng[1]:.2f} for all {len(variants)} variants (LODO datasets: held-out models)",
                  fontsize=11, color=INK, loc="left")

    lodo_rng = (min(s_main[(v, "width_ratio")][0] for v in variants), max(s_main[(v, "width_ratio")][0] for v in variants))
    caption = "\n".join([
        "Left: mean labelled vessel width at the labelled centreline divided by the FOV diameter (×1000), one dot per "
        f"image, bar = dataset mean. Relative to eye size, FIVES vessels are drawn {ratio_fh:.1f}× as wide as HRF's "
        f"({rel_mean['FIVES']:.2f} vs {rel_mean['HRF']:.2f}). In pixels the picture differs: FIVES {px_mean['FIVES']:.1f} "
        f"px, HRF {px_mean['HRF']:.1f}, DRIVE {px_mean['DRIVE']:.1f}, because pixel width also follows image resolution "
        "and eye size in pixels (542 px in DRIVE, 3376 px in HRF); only the relative width reflects the labelling "
        "convention.",
        "Right: width_ratio = predicted / labelled vessel width, measured at the labelled centreline points the model "
        "found (so it compares widths of vessels the model did detect, not missed vessels). Marker = 3-seed mean of the "
        "dataset mean, bar = range over seeds. Settings as in the Labelling-style test section of REPORT.md "
        "(label_style_test.py): frozen models, no training, no TTA, threshold 0.5, 3 seeds; LODO models on their own "
        "held-out dataset, models trained on all 113 pooled images on FIVES (792 images).",
        f"The thinning is shared: width_ratio is {lodo_rng[0]:.2f}–{lodo_rng[1]:.2f} on LODO (mean of the four datasets) "
        f"and {fives_rng[0]:.2f}–{fives_rng[1]:.2f} on FIVES for every variant, learned or not. It does not separate "
        f"the models: hybrid's extra FIVES loss against rgb is missed vessels (centreline recall {clr:+.3f}), not "
        f"thinner ones (width_ratio {s_sec[('hybrid', 'rgb')]['width_ratio']:+.3f}).",
        ("hybrid_weber is not in the labelling-style results (the test was run before it existed); variants shown: "
         + ", ".join(variants) + "." if "hybrid_weber" in absent else "Variants shown: " + ", ".join(variants) + ".")
        + (f" Not in the file: {', '.join(v for v in absent if v != 'hybrid_weber')}." if
           [v for v in absent if v != "hybrid_weber"] else ""),
    ])
    wrapped = "\n".join(textwrap.fill(p, 180) for p in caption.split("\n"))
    n_lines = wrapped.count("\n") + 1
    extra = 0.155 * n_lines + 0.2
    fig.set_figheight(fig.get_figheight() + extra)
    bottom = extra / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1), w_pad=3)
    fig.text(0.01, bottom - 0.01, wrapped, ha="left", va="top", fontsize=8.5, color=INK)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "label_style_widths.png", dpi=150)
    plt.close(fig)
    (out / "label_style_widths_caption.md").write_text(caption + "\n")
    print(f"FIVES/HRF relative width {ratio_fh:.2f}; FIVES width_ratio {fives_rng[0]:.3f}-{fives_rng[1]:.3f}; "
          f"hybrid-rgb cl_recall {clr:+.4f}; absent: {absent}")
    print(f"written to {out}/label_style_widths.png")


if __name__ == "__main__":
    main()
