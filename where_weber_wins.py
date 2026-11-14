"""EXPLORATORY (begun after seeing the paired-test result; nothing here is confirmatory). Rule: PROTOCOL.md,
"Exploratory: on which images does hybrid_weber beat rgb?", fixed 2026-11-13 before computing.

On which images does hybrid_weber beat rgb? No model is run: inputs are the existing per_image.csv files and the
images, labels and metadata in data/pool and data/external_FIVES (blank-label images left out: LODO 113, FIVES 792).

1. distribution of hybrid_weber - rgb per-image Dice (each seed and 3-seed per-image mean, LODO and FIVES)
2. association with fixed properties: degradation_test.measure (brightness, bright_outlier, vessel_contrast, noise,
   sharpness) and metadata (fov_diameter, vessel_fraction_in_fov) by Spearman within each dataset, LODO pooled and
   FIVES (42 tests); dataset (LODO) and condition (HRF, FIVES) by per-category median + Kruskal-Wallis (3 tests)
3. scatter of the 3-seed mean difference against the two properties with the largest mean |rho| over LODO pooled
   and FIVES

Outputs (results/where_weber_wins/): summary.md, measures.csv (cached image measures), differences.csv,
correlations.csv, scatter.png

Usage:
    python where_weber_wins.py [--workers 6]
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kruskal, spearmanr

from dataset_gap_final import COLORS5
from paired_tests import SEEDS, load
from u2nete.data import load_metadata
from u2nete.io import read_binary, read_rgb
from visualize_enhancement import GRID, INK, MUTED, style

A, B = "hybrid_weber", "rgb"
TRAIN = ["CHASE_DB1", "DRIVE", "HRF", "STARE"]
POOLS = {"LODO": "data/pool", "FIVES": "data/external_FIVES"}
MEASURES = ["brightness", "bright_outlier", "vessel_contrast", "noise", "sharpness"]
META = ["fov_diameter", "vessel_fraction_in_fov"]
CONTINUOUS = MEASURES + META
GROUPS = [*TRAIN, "LODO pooled", "FIVES"]
N_TESTS = len(CONTINUOUS) * len(GROUPS) + 3
THRESH = [0.02, 0.05, 0.10]

# Written after the first run, from the numbers below; exploratory reading only.
READING = "\n".join([
    "**LODO: a few large gains, not many small ones.** The positive mean (+0.0055) against a negative median "
    "(−0.0015) comes from three STARE images: im0324 (+0.388), im0319 (+0.344) and im0004 (+0.205), each large in "
    "every seed. On them `rgb` partly fails (Dice 0.38–0.61 across seeds) while `hybrid_weber` stays at 0.78–0.80. "
    "Without those three images the mean is −0.0029 and the median −0.0016. Apart from them, gains and losses are "
    "small and balanced (22 images gain and 21 lose more than 0.02). Two of the three are among the brightest images "
    "in the pool (green mean 131–132, STARE median 103).",
    "",
    "**LODO: no shared property within datasets.** No property passes the Bonferroni reference within a dataset "
    "except sharpness in CHASE_DB1 (ρ +0.60), and the signs disagree between datasets (noise +0.38 in CHASE_DB1, "
    "−0.45 in STARE; bright_outlier +0.42 in DRIVE, −0.34 in HRF). The strong pooled correlations (fov_diameter "
    "+0.53, vessel fraction −0.48, sharpness +0.45, brightness −0.38) follow the dataset: `hybrid_weber` wins on 73% "
    "of HRF images (large eyes) and on none of DRIVE's (Kruskal–Wallis p 3e-6), so they describe datasets, not images "
    "within a dataset. In HRF the gain is on the diseased eyes (diabetic retinopathy 87% won, glaucoma 93%) and not "
    "the healthy ones (40%).",
    "",
    "**FIVES: many small gains, ordered by brightness.** 589 of 792 images gain, and the 10% largest gains supply "
    "only 30% of the total, so the FIVES mean is a broad shift, not a few outliers. Brightness is the strongest "
    "association in the whole analysis (ρ +0.76): `hybrid_weber` beats `rgb` on 95–98% of the brighter half and on "
    "23% of the darkest quarter (green mean below 37.5). Its correlates point the same way (bright_outlier −0.71 and "
    "relative noise −0.60 are both higher in dark images), and glaucoma, the darkest condition (median green 35.7 "
    "against 46.2), is the only one it loses (40% won). (Added after seeing the numbers, descriptive only: the "
    "brightness ρ is positive in each seed, +0.61 / +0.47 / +0.80, so seed 2's overall loss shifts the whole curve "
    "down rather than reversing it.)",
    "",
    "**Same pattern on LODO and FIVES? No.** On FIVES the advantage grows steadily with brightness. On LODO there is "
    "no such gradient within any dataset (brightness ρ −0.28 to +0.10), and the pooled sign is the opposite (−0.38, "
    "a dataset effect). The one point of contact is that the three large LODO wins include the two brightest pool "
    "images, which fits the FIVES direction, but three images cannot carry a pattern. All of this is exploratory: "
    "the brightness link is a hypothesis for a pre-registered test on new data, not a finding.",
])


def _measure(args):
    from degradation_test import measure  # imported in the worker (spawn)
    pool, i, diam = args
    rgb = read_rgb(Path(pool) / "images" / f"{i}.png")
    gt = read_binary(Path(pool) / "masks" / f"{i}.png")
    fov = read_binary(Path(pool) / "fov" / f"{i}.png")
    m = measure(rgb, gt, fov, diam)
    return i, {k: m[k] for k in MEASURES}


def image_measures(ids_by_setting, meta, cache, workers):
    have = {}
    if cache.exists():
        with open(cache) as fh:
            have = {r["id"]: {k: float(r[k]) for k in MEASURES} for r in csv.DictReader(fh)}
    todo = [(POOLS[s], i, float(meta[i]["fov_diameter"])) for s, ids in ids_by_setting.items() for i in ids
            if i not in have]
    if todo:
        print(f"measuring {len(todo)} images ({workers} workers)")
        with Pool(workers) as p:
            for n, (i, m) in enumerate(p.imap_unordered(_measure, todo, chunksize=4)):
                have[i] = m
                if (n + 1) % 100 == 0:
                    print(f"  {n + 1}/{len(todo)}")
        with open(cache, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", *MEASURES])
            w.writeheader()
            for i in sorted(have):
                w.writerow({"id": i, **have[i]})
    return have


def dist_rows(d):
    q = np.percentile(d, np.arange(10, 100, 10))
    row = {"n": len(d), "median": float(np.median(d)), "mean": float(np.mean(d)), "deciles": q.tolist()}
    for t in THRESH:
        row[f"gain>{t}"] = int((d > t).sum())
        row[f"loss>{t}"] = int((d < -t).sum())
    pos, neg = d[d > 0], d[d < 0]
    row["n_pos"], row["n_neg"] = len(pos), len(neg)
    row["sum_pos"], row["sum_neg"] = float(pos.sum()), float(neg.sum())
    # share of the total gain contributed by the largest 10% of gaining images
    top = np.sort(pos)[::-1][:max(1, int(np.ceil(0.1 * len(pos))))] if len(pos) else np.array([0.0])
    row["top10pct_share_of_gain"] = float(top.sum() / pos.sum()) if len(pos) else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/where_weber_wins")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- differences per seed and 3-seed mean
    diffs = {}
    for s_name in POOLS:
        da = {s: load(runs, s_name, A, s) for s in SEEDS}
        db = {s: load(runs, s_name, B, s) for s in SEEDS}
        ids = sorted(da[1])
        assert all(set(da[s]) == set(ids) == set(db[s]) for s in SEEDS)
        diffs[s_name] = {"ids": ids, **{f"seed {s}": np.array([da[s][i] - db[s][i] for i in ids]) for s in SEEDS}}
        diffs[s_name]["3-seed mean"] = np.mean([diffs[s_name][f"seed {s}"] for s in SEEDS], axis=0)
    assert len(diffs["LODO"]["ids"]) == 113 and len(diffs["FIVES"]["ids"]) == 792

    meta = {**load_metadata(POOLS["LODO"]), **load_metadata(POOLS["FIVES"])}
    meas = image_measures({k: v["ids"] for k, v in diffs.items()}, meta, out / "measures.csv", args.workers)

    # one table of images: id, setting, dataset, condition, properties, 3-seed mean diff
    rows = []
    for s_name, d in diffs.items():
        for n, i in enumerate(d["ids"]):
            m = meta[i]
            rows.append({"id": i, "setting": s_name, "dataset": m["dataset"], "condition": m.get("condition", ""),
                         **meas[i], "fov_diameter": float(m["fov_diameter"]),
                         "vessel_fraction_in_fov": float(m["vessel_fraction_in_fov"]),
                         **{f"diff_{k}": float(d[k][n]) for k in [*[f"seed {s}" for s in SEEDS], "3-seed mean"]}})
    with open(out / "differences.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def group(g):
        if g == "LODO pooled":
            return [r for r in rows if r["setting"] == "LODO"]
        return [r for r in rows if r["dataset"] == g]

    # --- correlations
    corr = []
    for prop in CONTINUOUS:
        for g in GROUPS:
            rs = group(g)
            x, y = np.array([r[prop] for r in rs]), np.array([r["diff_3-seed mean"] for r in rs])
            ok = np.isfinite(x)
            rho, p = spearmanr(x[ok], y[ok])
            corr.append({"property": prop, "group": g, "n": int(ok.sum()), "rho": float(rho), "p": float(p)})
    with open(out / "correlations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(corr[0]))
        w.writeheader()
        w.writerows(corr)
    cats = []
    for label, rs, key in [("dataset, LODO pooled", group("LODO pooled"), "dataset"),
                           ("condition, HRF", group("HRF"), "condition"),
                           ("condition, FIVES", group("FIVES"), "condition")]:
        levels = sorted({r[key] for r in rs})
        vals = [np.array([r["diff_3-seed mean"] for r in rs if r[key] == lv]) for lv in levels]
        cats.append({"label": label, "levels": levels, "n": [len(v) for v in vals],
                     "median": [float(np.median(v)) for v in vals], "mean": [float(np.mean(v)) for v in vals],
                     "frac_win": [float((v > 0).mean()) for v in vals], "p": float(kruskal(*vals).pvalue)})

    strength = {p: np.mean([abs(c["rho"]) for c in corr if c["property"] == p and c["group"] in ("LODO pooled", "FIVES")])
                for p in CONTINUOUS}
    top2 = sorted(CONTINUOUS, key=lambda p: -strength[p])[:2]

    # --- figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")
    for ax, prop in zip(axes, top2):
        ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
        for ds in [*TRAIN, "FIVES"]:
            rs = [r for r in rows if r["dataset"] == ds]
            big = ds != "FIVES"
            ax.scatter([r[prop] for r in rs], [r["diff_3-seed mean"] for r in rs], s=30 if big else 10,
                       color=COLORS5[ds], alpha=0.95 if big else 0.45, edgecolor="white" if big else "none",
                       linewidth=0.8, label=f"{ds} (n={len(rs)})", zorder=3 if big else 2)
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        rl = next(c for c in corr if c["property"] == prop and c["group"] == "LODO pooled")
        rf = next(c for c in corr if c["property"] == prop and c["group"] == "FIVES")
        ax.set_title(f"{prop}\nSpearman ρ: LODO pooled {rl['rho']:+.2f} (p {rl['p']:.2g}) · FIVES {rf['rho']:+.2f} "
                     f"(p {rf['p']:.2g})", fontsize=10.5, color=INK)
        ax.set_xlabel(prop, fontsize=9.5, color=MUTED)
    axes[0].set_ylabel(f"{A} − {B} Dice (3-seed per-image mean)", fontsize=9.5, color=MUTED)
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="best", markerscale=1.2)
    fig.suptitle(f"EXPLORATORY: where {A} beats {B} (above 0) against the two properties most associated with the "
                 f"difference; {N_TESTS} tests in all, p not corrected", fontsize=10.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "scatter.png", dpi=140)
    plt.close(fig)

    # --- summary
    L = [f"# EXPLORATORY: on which images does `{A}` beat `{B}`?", "",
         "**Exploratory, not confirmatory.** Begun after seeing the paired-test result (results/paired_tests); the "
         "analysis was fixed in PROTOCOL.md before computing. No claim here is confirmatory; a pattern found here "
         "would need a new pre-registered test on new data. No model was run.", "",
         f"Difference = `{A}` − `{B}` per-image Dice (8-fold TTA, threshold 0.5). LODO: 113 images, each scored by "
         "the fold that held its dataset out. FIVES: 792 images with a non-blank label.", "",
         "## 1. Distribution of the difference", ""]
    dec_head = " | ".join(f"P{q}" for q in range(10, 100, 10))
    for s_name in POOLS:
        L += [f"### {s_name}", "", f"| analysis | n | median | mean | {dec_head} |", "|---|---|---|---|" + "---|" * 9]
        drs = {k: dist_rows(diffs[s_name][k]) for k in [*[f"seed {s}" for s in SEEDS], "3-seed mean"]}
        for k, r in drs.items():
            L.append(f"| {k} | {r['n']} | {r['median']:+.4f} | {r['mean']:+.4f} | "
                     + " | ".join(f"{x:+.3f}" for x in r["deciles"]) + " |")
        L += ["", "| analysis | images gaining / losing | " + " | ".join(f"gain > {t} / loss > {t}" for t in THRESH)
              + " | total gain / total loss | top 10% of gaining images: share of total gain |",
              "|---|---|" + "---|" * len(THRESH) + "---|---|"]
        for k, r in drs.items():
            L.append(f"| {k} | {r['n_pos']} / {r['n_neg']} | "
                     + " | ".join(f"{r[f'gain>{t}']} / {r[f'loss>{t}']}" for t in THRESH)
                     + f" | {r['sum_pos']:+.2f} / {r['sum_neg']:+.2f} | {r['top10pct_share_of_gain']:.0%} |")
        L.append("")

    L += ["## 2. Association with image properties (3-seed per-image mean difference)", "",
          f"{N_TESTS} tests in all (42 Spearman + 3 Kruskal–Wallis). p-values are not corrected; Bonferroni reference "
          f"0.05 / {N_TESTS} = {0.05 / N_TESTS:.4f}, for orientation only. ρ > 0: higher values of the property go with "
          f"`{A}` doing better relative to `{B}`.", "",
          "### Spearman ρ (p), by group", "",
          "| property | " + " | ".join(f"{g} (n={len(group(g))})" for g in GROUPS) + " |",
          "|---|" + "---|" * len(GROUPS)]
    for prop in CONTINUOUS:
        cells = []
        for g in GROUPS:
            c = next(c for c in corr if c["property"] == prop and c["group"] == g)
            mark = "**" if c["p"] < 0.05 / N_TESTS else ""
            cells.append(f"{mark}{c['rho']:+.2f}{mark} ({c['p']:.2g})")
        L.append(f"| {prop} | " + " | ".join(cells) + " |")
    L += ["", f"Bold: p < {0.05 / N_TESTS:.4f} (Bonferroni over {N_TESTS} tests).", "",
          "### Categorical properties (median difference, mean difference, fraction of images won, by category)", "",
          "| property | category | n | median | mean | images won | Kruskal–Wallis p |", "|---|---|---|---|---|---|---|"]
    for c in cats:
        for j, lv in enumerate(c["levels"]):
            L.append(f"| {c['label']} | {lv or '(none)'} | {c['n'][j]} | {c['median'][j]:+.4f} | {c['mean'][j]:+.4f} | "
                     f"{c['frac_win'][j]:.0%} | {f'{c['p']:.2g} ({N_TESTS} tests)' if j == 0 else ''} |")
    L += ["", f"Figure: scatter.png, the difference against the two properties with the largest mean |ρ| over LODO "
          f"pooled and FIVES ({top2[0]}: {strength[top2[0]]:.2f}; {top2[1]}: {strength[top2[1]]:.2f}).", ""]

    L += ["## 3. Reading (exploratory)", "", READING or "(to be written from the numbers above)", ""]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
