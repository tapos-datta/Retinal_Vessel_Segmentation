"""Robustness figure from the degradation test (existing CSV files; no model is run).

Data: results/degradation/per_image.csv (rgb, clahe, hybrid, enh3) and results/degradation_stage6b/per_image.csv
(hybrid_weber), 3 seeds each, frozen LODO models on their own 113 held-out images, no TTA, threshold 0.5.

  main panel   mean Dice per level (mean over seeds of the per-seed mean over images), seed range shaded; levels in
               degradation_test.LEVELS order, grouped by family (lines join levels within a family only); the two rule
               levels (disc_32, fives) are marked
  side strip   one image (results/degradation/examples) at original, disc_32 and fives

Every drawn mean is checked against the "Mean Dice by level" tables of results/degradation/summary.md and
results/degradation_stage6b/summary.md (4 decimals); any mismatch stops the script. A variant missing at a level is
named in the caption. The ranking sentence in the caption is computed, against the FIVES 3-seed means from the run
folders (stage6b_summary.fives).

Outputs (figures/report/): degradation_robustness.png, degradation_robustness_caption.md

Usage:
    python degradation_figure.py
"""
import argparse
import csv
import statistics as st
import textwrap
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from degradation_test import LEVELS, ORIGINAL, PRIMARY
from stage6b_summary import SEEDS, fives
from visualize_enhancement import GRID, INK, MUTED, style

VARIANTS = ["rgb", "clahe", "hybrid", "enh3", "hybrid_weber"]
COLORS = {"rgb": "#2a78d6", "clahe": "#eb6834", "hybrid": "#1baf7a", "enh3": "#eda100", "hybrid_weber": "#e87ba4"}
SOURCES = ["results/degradation/per_image.csv", "results/degradation_stage6b/per_image.csv"]
SUMMARIES = ["results/degradation/summary.md", "results/degradation_stage6b/summary.md"]
RULE_LEVELS = ["disc_32", PRIMARY]
FAMILIES = [[ORIGINAL], [l for l in LEVELS if l.startswith("disc_")], [l for l in LEVELS if l.startswith("contrast_")],
            [l for l in LEVELS if l.startswith("noise_")], [PRIMARY, "strong"]]
EXAMPLE = "results/degradation/examples/HRF_01_dr.png"
EXAMPLE_LEVELS = [ORIGINAL, "disc_32", PRIMARY]
TILE_W, LABEL_STRIP = 360, 30  # save_examples: 360 px tiles, 4 per row, level name written in the top ~30 px


def load(paths):
    """{(variant, level): {seed: [dice, ...]}}; a variant/seed/level found in two files is an error."""
    data, origin = defaultdict(lambda: defaultdict(list)), {}
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                key = (r["variant"], r["seed"], r["level"])
                if origin.setdefault(key, p) != p:
                    raise SystemExit(f"{key} appears in both {origin[key]} and {p}")
                data[(r["variant"], r["level"])][r["seed"]].append(float(r["dice"]))
    return data


def reference(paths):
    """{(variant, level): mean} from the 'Mean Dice by level' tables; a value given twice must agree."""
    ref = {}
    for p in paths:
        sec = Path(p).read_text().split("## Mean Dice by level")[1].split("\n## ")[0]
        lines = [l for l in sec.splitlines() if l.startswith("|")]
        head = [c.strip() for c in lines[0].strip("|").split("|")]
        for l in lines[2:]:
            cells = [c.strip() for c in l.strip("|").split("|")]
            for v in VARIANTS:
                if v in head:
                    val = float(cells[head.index(v)])
                    k = (v, cells[0])
                    if k in ref and abs(ref[k] - val) > 1e-9:
                        raise SystemExit(f"{k}: {p} has {val}, an earlier summary has {ref[k]}")
                    ref[k] = val
    return ref


def tiles(path, levels):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    th = img.shape[0] // -(-len(LEVELS) // 4)
    out = []
    for lv in levels:
        k = LEVELS.index(lv)
        y, x = (k // 4) * th, (k % 4) * TILE_W
        out.append(img[y + LABEL_STRIP:y + th, x:x + TILE_W])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures/report")
    args = ap.parse_args()

    data = load(SOURCES)
    ref = reference(SUMMARIES)
    missing, mean, lo, hi = [], {}, {}, {}
    for v in VARIANTS:
        for lv in LEVELS:
            seeds = data.get((v, lv))
            if not seeds:
                missing.append(f"{v} at {lv}")
                continue
            per_seed = [st.mean(x) for x in seeds.values()]
            mean[(v, lv)], lo[(v, lv)], hi[(v, lv)] = st.mean(per_seed), min(per_seed), max(per_seed)
            if len(seeds) != 3 or len({len(x) for x in seeds.values()}) != 1:
                missing.append(f"{v} at {lv}: {len(seeds)} seeds, images per seed {sorted({len(x) for x in seeds.values()})}")

    errors = []
    for k, m in mean.items():
        if k not in ref:
            errors.append(f"{k}: drawn but not in the summaries")
        elif abs(round(m, 4) - ref[k]) > 1e-9:
            errors.append(f"{k}: drawn {m:.4f}, summary {ref[k]:.4f}")
    if errors:
        raise SystemExit("drawn values differ from the summaries:\n  " + "\n  ".join(errors))
    print(f"all {len(mean)} drawn means match {' and '.join(SUMMARIES)}")

    # ranking at the FIVES-matched level vs on FIVES itself (3-seed means from the run folders, 792 images)
    fv = {v: st.mean(fives(args.runs, v, s) for s in SEEDS) for v in VARIANTS}
    order = lambda d: " > ".join(f"{v}" for v in sorted(VARIANTS, key=lambda v: -round(d[v], 4)))
    deg = {v: mean[(v, PRIMARY)] for v in VARIANTS if (v, PRIMARY) in mean}
    four = [v for v in VARIANTS if v != "hybrid_weber"]
    # reversed: every pair of the four is ordered oppositely (a tie on either side counts as compatible)
    sgn = lambda d, a, b: np.sign(round(d[a], 4) - round(d[b], 4))
    reversed4 = all(sgn(deg, a, b) * sgn(fv, a, b) <= 0 for i, a in enumerate(four) for b in four[i + 1:])
    ties =[(a, b) for i, a in enumerate(four) for b in four[i + 1:] if round(fv[a], 4) == round(fv[b], 4)]

    # --- figure
    fig = plt.figure(figsize=(13, 5.4), facecolor="white")
    gs = fig.add_gridspec(len(EXAMPLE_LEVELS), 2, width_ratios=[4.3, 1], wspace=0.06, hspace=0.32)
    ax = fig.add_subplot(gs[:, 0])
    xpos, x, gap = {}, 0.0, 0.7
    for fam in FAMILIES:
        for lv in fam:
            xpos[lv] = x
            x += 1
        x += gap
    for lv in RULE_LEVELS:
        ax.axvspan(xpos[lv] - 0.38, xpos[lv] + 0.38, color="#f1f0ea", zorder=0)
        ax.text(xpos[lv], 1.0, "rule level", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8.5, color=MUTED)
    for j, v in enumerate(VARIANTS):
        for fam in FAMILIES:
            pts = [lv for lv in fam if (v, lv) in mean]
            # the single original point: spread the variants sideways so their seed ranges do not overlap
            xs = [xpos[lv] + ((j - 2) * 0.09 if len(fam) == 1 else 0) for lv in pts]
            ax.fill_between(xs, [lo[(v, lv)] for lv in pts], [hi[(v, lv)] for lv in pts], color=COLORS[v],
                            alpha=0.14, linewidth=0, zorder=2)
            if len(pts) == 1:
                ax.vlines(xs, [lo[(v, pts[0])]], [hi[(v, pts[0])]], color=COLORS[v], alpha=0.35, linewidth=4, zorder=2)
            ax.plot(xs, [mean[(v, lv)] for lv in pts], color=COLORS[v], linewidth=2, marker="o", markersize=6,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3,
                    label=v if fam is FAMILIES[0] else None)
        last = FAMILIES[-1][-1]
        if (v, last) in mean:
            ax.annotate(f"{v}  {mean[(v, last)]:.3f}", (xpos[last], mean[(v, last)]), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
    style(ax)
    ax.set_xticks([xpos[lv] for lv in LEVELS], LEVELS, rotation=35, ha="right", fontsize=9, color=INK)
    ax.set_xlim(-0.5, x - gap + 1.6)
    ax.set_ylabel("Dice (mean over 113 images; 3-seed mean, seed range shaded)", fontsize=9.5, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left", ncol=5)
    ax.set_title("Dice under calibrated image degradation (frozen LODO models, no TTA)", fontsize=11.5,
                 color=INK, loc="left", pad=16)

    for r, (lv, t) in enumerate(zip(EXAMPLE_LEVELS, tiles(EXAMPLE, EXAMPLE_LEVELS))):
        a = fig.add_subplot(gs[r, 1])
        a.imshow(t)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title(("HRF 01_dr · " if r == 0 else "") + lv, fontsize=9.5, color=INK, pad=3, loc="left")

    tie_txt = (" (" + ", ".join(f"{a} and {b} tie at 4 decimals" for a, b in ties) + ")") if ties else ""
    rank_txt = (f"At the FIVES-matched level `{PRIMARY}` the order is {order(deg)}; on FIVES itself (3-seed means, "
                f"792 images) it is {order(fv)}{tie_txt}. ")
    if reversed4:
        rank_txt += ("Among rgb, clahe, hybrid and enh3 the ranking here is the reverse of the FIVES ranking, so the "
                     "degradation does not reproduce what goes wrong on FIVES; hybrid_weber is first in both.")
    else:
        rank_txt += "The ranking among rgb, clahe, hybrid and enh3 is not a simple reversal of the FIVES ranking."
    caption = "\n".join([
        "Mean Dice per degradation level for frozen LODO models on their own 113 held-out images; line = mean over "
        "seeds 1–3 of the per-seed mean, shading = range of the three seed means. Lines join levels within one family "
        "(tone with a bright optic disc, haze lowering vessel contrast, noise, and the matched chains). No TTA in the "
        "degradation test, threshold 0.5, Dice inside the FOV at original resolution.",
        "Levels are calibrated per image: each step moves a property of that image by the factor separating the "
        "darkest FIVES quarter from the training images, and `fives` combines them (see the Degradation test section "
        "of REPORT.md); 52 images reach the `fives` targets only partly. Shaded columns: disc_32 and fives, the "
        "levels used by the stage 6 and 6b robustness rules. Right: one image at original, disc_32 and fives.",
        rank_txt,
        "Data: results/degradation (rgb, clahe, hybrid, enh3) and results/degradation_stage6b (hybrid_weber), 3 "
        "seeds each; " + ("all variants present at every level." if not missing else
                          "missing or incomplete: " + "; ".join(missing) + "."),
    ])
    wrapped = "\n".join(textwrap.fill(p, 175) for p in caption.split("\n"))
    n_lines = wrapped.count("\n") + 1
    extra = 0.155 * n_lines + 0.2
    fig.set_figheight(fig.get_figheight() + extra)
    bottom = extra / fig.get_figheight()
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=bottom + 0.11)
    fig.text(0.01, bottom - 0.01, wrapped, ha="left", va="top", fontsize=8.5, color=INK)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "degradation_robustness.png", dpi=150)
    plt.close(fig)
    (out / "degradation_robustness_caption.md").write_text(caption + "\n")
    print(rank_txt)
    print(f"written to {out}/degradation_robustness.png")


if __name__ == "__main__":
    main()
