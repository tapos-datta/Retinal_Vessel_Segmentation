"""Headline results chart: LODO mean Dice (left) and FIVES Dice (right) for the five Part 1 variants.

Both panels show the same variants in the same order, the 3-seed mean as a short horizontal bar and each seed as a
dot, on one shared y-axis, with each variant's rank within the panel written above it: the point of the figure is that
the two evaluations rank the variants differently.

Numbers come from the run folders through stage6b_summary.lodo / stage6b_summary.fives (LODO: per seed, the mean of
the four held-out dataset means; FIVES: per seed, the mean over the 792 scorable images, blank labels excluded). Every
drawn value (mean and each seed, rounded to 4 decimals) is checked against results/stage6b/summary.md; any difference
stops the script before anything is written.

Outputs (figures/report/): results_lodo_vs_fives.png, results_lodo_vs_fives_caption.md

Usage:
    python results_charts.py
"""
import argparse
import re
import statistics as st
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stage6b_summary import SEEDS, fives, lodo
from visualize_enhancement import GRID, INK, MUTED

VARIANTS = ["rgb", "clahe", "hybrid", "enh3", "hybrid_weber"]
PANELS = [("lodo", "LODO: mean Dice over the 4 held-out datasets"), ("fives", "FIVES: Dice over 792 external images")]


def collect(runs):
    """{panel: {variant: [seed1, seed2, seed3]}} from the run folders."""
    out = {"lodo": {}, "fives": {}}
    for v in VARIANTS:
        per = [lodo(runs, v, s) for s in SEEDS]
        if None in per:
            raise SystemExit(f"{v}: a LODO fold is missing in {runs}")
        out["lodo"][v] = [st.mean(p.values()) for p in per]
        fv = [fives(runs, v, s) for s in SEEDS]
        if None in fv:
            raise SystemExit(f"{v}: a FIVES evaluation is missing in {runs}")
        out["fives"][v] = fv
    return out


def reference(path):
    """{panel: {variant: (mean, [seeds])}} parsed from the stage6b summary.md tables."""
    text = Path(path).read_text()
    sections = {"lodo": text.split("## LODO")[1].split("##")[0], "fives": text.split("## FIVES")[1].split("##")[0]}
    ref = {}
    for panel, sec in sections.items():
        ref[panel] = {}
        for line in sec.splitlines():
            m = re.match(r"\| `([^`]+)` \|(.*)\|\s*$", line)
            if not m:
                continue
            cells = [c.strip() for c in m.group(2).split("|")]
            seeds = [float(x) for x in cells[-1].split("/")]
            ref[panel][m.group(1)] = (float(cells[-2]), seeds)
    return ref


def check(data, ref, ref_path):
    errors = []
    for panel, by_v in data.items():
        for v, seeds in by_v.items():
            if v not in ref[panel]:
                errors.append(f"{panel} {v}: not in {ref_path}")
                continue
            rmean, rseeds = ref[panel][v]
            drawn = [round(st.mean(seeds), 4), *[round(x, 4) for x in seeds]]
            want = [rmean, *rseeds]
            for name, a, b in zip(["mean", *[f"seed {s}" for s in SEEDS]], drawn, want):
                if abs(a - b) > 1e-9:
                    errors.append(f"{panel} {v} {name}: drawn {a:.4f} but {ref_path} has {b:.4f}")
    if errors:
        raise SystemExit("values differ from the reference summary:\n  " + "\n  ".join(errors))
    print(f"all {sum(len(b) for b in data.values()) * (1 + len(SEEDS))} drawn values match {ref_path}")


def ranks(means):
    """Rank 1 = highest mean (4-decimal values, as tabulated); equal values share a rank, shown as '=2'."""
    r4 = {v: round(m, 4) for v, m in means.items()}
    out = {}
    for v, m in r4.items():
        rank = 1 + sum(x > m for x in r4.values())
        tied = sum(x == m for x in r4.values()) > 1
        out[v] = f"{'=' if tied else ''}{rank}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--reference", default="results/stage6b/summary.md")
    ap.add_argument("--out", default="figures/report")
    args = ap.parse_args()

    data = collect(args.runs)
    check(data, reference(args.reference), args.reference)
    means = {p: {v: st.mean(s) for v, s in data[p].items()} for p in data}
    rk = {p: ranks(means[p]) for p in data}

    allv = [x for p in data for s in data[p].values() for x in s]
    lo, hi = np.floor(min(allv) * 100 - 1) / 100, np.ceil(max(allv) * 100 + 1) / 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True, facecolor="white")
    xs = np.arange(len(VARIANTS))
    for ax, (panel, title) in zip(axes, PANELS):
        base = means[panel]["rgb"]
        ax.axhline(base, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        for x, v in zip(xs, VARIANTS):
            seeds = data[panel][v]
            jitter = np.array([-0.09, 0.0, 0.09])
            ax.scatter(x + jitter, seeds, s=36, color="#8f8e86", edgecolor="white", linewidth=1.2, zorder=3)
            m = means[panel][v]
            ax.plot([x - 0.26, x + 0.26], [m, m], color=INK, linewidth=3, solid_capstyle="round", zorder=4)
            ax.annotate(f"{m:.3f}", (x + 0.3, m), xytext=(2, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
            ax.annotate((rk[panel][v] if rk[panel][v].startswith("=") else f"#{rk[panel][v]}"), (x, max(seeds)), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
        ax.set_title(title, fontsize=11.5, color=INK, pad=10, loc="left")
        ax.set_xticks(xs, VARIANTS, fontsize=10, color=INK)
        ax.set_xlim(-0.55, len(VARIANTS) - 0.35)
        ax.set_ylim(lo, hi)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Dice", fontsize=10, color=MUTED)

    order = lambda p: " > ".join(sorted(VARIANTS, key=lambda v: -round(means[p][v], 4)))
    caption = (
        "Dice of the five Part 1 variants, same order in both panels. Bar = 3-seed mean (value at its right); dots = "
        "seeds 1, 2, 3; #n = rank of the mean within the panel (=n marks a tie at 4 decimals); dashed line = rgb mean. "
        "Both panels share one y-axis. Left: leave-one-dataset-out, per seed the mean of the four held-out dataset "
        "means (CHASE_DB1, DRIVE, HRF, STARE). Right: FIVES, models trained on all 113 pooled images, mean over the "
        "792 images with a non-blank label. 8-fold TTA, threshold 0.5, Dice inside the FOV at original resolution. "
        f"LODO order: {order('lodo')}. FIVES order: {order('fives')}. Values from results/stage6b/summary.md, "
        "recomputed from the run folders."
    )
    wrapped = textwrap.fill(caption, 150)
    n_lines = wrapped.count("\n") + 1
    bottom = (0.16 * n_lines + 0.12) / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1), w_pad=3)
    fig.text(0.01, bottom - 0.01, wrapped, ha="left", va="top", fontsize=8.5, color=INK)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "results_lodo_vs_fives.png", dpi=150)
    plt.close(fig)
    (out / "results_lodo_vs_fives_caption.md").write_text(caption + "\n")
    for p in data:
        print(f"{p:5}: " + "  ".join(f"{v} {means[p][v]:.4f} (#{rk[p][v]})" for v in VARIANTS))
    print(f"written to {out}/results_lodo_vs_fives.png")


if __name__ == "__main__":
    main()
