"""Stage 6b decision: rules C1, C2 and C3 from PROTOCOL.md for hybrid_weber over 3 seeds.

  C1  FIVES 3-seed mean Dice >= the rgb 3-seed mean, and above rgb of the same seed in at least 2 of 3 seeds
  C2  LODO 3-seed mean Dice (mean of the 4 held-out dataset means) >= the rgb 3-seed mean
  C3  Dice lost under degradation (no TTA) at disc_32 and fives no more than hybrid's + 0.010, 3 seeds

Usage: python stage6b_summary.py [--out results/stage6b]
Missing runs are shown as "–" and the verdict as pending.
"""
import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

V = "hybrid_weber"
DATASETS = ["CHASE_DB1", "DRIVE", "HRF", "STARE"]
SEEDS = (1, 2, 3)
C3_LEVELS, C3_MARGIN = ("disc_32", "fives"), 0.010


def lodo(runs, variant, seed):
    """{dataset: mean Dice} for one seed, or None if a fold is missing."""
    out = {}
    for d in DATASETS:
        f = Path(runs) / variant / "lodo" / f"lodo_{d}" / f"seed{seed}" / "test" / "per_image.csv"
        if not f.exists():
            return None
        out[d] = st.mean(float(r["dice"]) for r in csv.DictReader(open(f)))
    return out


def fives(runs, variant, seed):
    f = Path(runs) / variant / "all" / "all" / f"seed{seed}" / "external_FIVES" / "per_image.csv"
    if not f.exists():
        return None
    return st.mean(float(r["dice"]) for r in csv.DictReader(open(f)) if r["auc"] != "nan")  # blank labels excluded


def losses(paths, variant):
    """({level: Dice - own original, over all seeds and images}, seeds present) for one variant."""
    d, seeds = defaultdict(list), set()
    for p in paths:
        if Path(p).exists():
            for r in csv.DictReader(open(p)):
                if r["variant"] == variant:
                    d[r["level"]].append(float(r["dice"]))
                    seeds.add(r["seed"])
    if "original" not in d:
        return {}, seeds
    return {lv: st.mean(x) - st.mean(d["original"]) for lv, x in d.items()
            if lv != "original" and len(x) == len(d["original"])}, seeds


def f4(x):
    return "–" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--degradation", default="results/degradation_stage6b/per_image.csv")
    ap.add_argument("--reference", default="results/degradation/per_image.csv")
    ap.add_argument("--out", default="results/stage6b")
    args = ap.parse_args()

    names = [V, "hybrid", "rgb", "clahe", "enh3"]
    L = ["# Stage 6b: hybrid_weber over 3 seeds", "", "## LODO (8-fold TTA)", "",
         "| Variant | " + " | ".join(DATASETS) + " | Mean | Seed 1 / 2 / 3 |", "|---|" + "---|" * (len(DATASETS) + 2)]
    lodo_mean, lodo_all = {}, {}
    for v in names:
        per = [lodo(args.runs, v, s) for s in SEEDS]
        lodo_all[v] = per
        got = [p for p in per if p]
        seedm = [st.mean(p.values()) if p else None for p in per]
        lodo_mean[v] = st.mean(x for x in seedm if x is not None) if len(got) == 3 else None
        cells = [f4(st.mean(p[d] for p in got)) if got else "–" for d in DATASETS]
        L.append(f"| `{v}` | " + " | ".join(cells) + f" | {f4(lodo_mean[v])} | " + " / ".join(f4(x) for x in seedm) + " |")
    if all(lodo_all[V]) and all(lodo_all["hybrid"]):
        L += ["", f"`{V}` − `hybrid` by held-out dataset and seed:", "", "| Seed | " + " | ".join(DATASETS) + " |",
              "|---|" + "---|" * len(DATASETS)]
        for s, a, b in zip(SEEDS, lodo_all[V], lodo_all["hybrid"]):
            L.append(f"| {s} | " + " | ".join(f"{a[d] - b[d]:+.4f}" for d in DATASETS) + " |")

    L += ["", "## FIVES (8-fold TTA, 792 images)", "", "| Variant | Mean | Seed 1 / 2 / 3 |", "|---|---|---|"]
    fv = {}
    for v in names:
        fv[v] = [fives(args.runs, v, s) for s in SEEDS]
        m = st.mean(fv[v]) if None not in fv[v] else None
        L.append(f"| `{v}` | {f4(m)} | " + " / ".join(f4(x) for x in fv[v]) + " |")

    lw, sw = losses([args.degradation], V)
    lh, _ = losses([args.reference], "hybrid")
    L += ["", "## Degradation test: Dice lost from own original images (no TTA)", "",
          "| Variant | " + " | ".join(C3_LEVELS) + " |", "|---|" + "---|" * len(C3_LEVELS),
          f"| `{V}` ({len(sw)} seeds) | " + " | ".join(f"{lw[lv]:+.4f}" if lv in lw else "–" for lv in C3_LEVELS) + " |",
          "| `hybrid` | " + " | ".join(f"{lh[lv]:+.4f}" if lv in lh else "–" for lv in C3_LEVELS) + " |"]

    c1 = c2 = c3 = None
    if None not in fv[V] and None not in fv["rgb"]:
        wins = sum(a > b for a, b in zip(fv[V], fv["rgb"]))
        c1 = st.mean(fv[V]) >= st.mean(fv["rgb"]) and wins >= 2
    if lodo_mean[V] is not None and lodo_mean["rgb"] is not None:
        c2 = lodo_mean[V] >= lodo_mean["rgb"]
    if len(sw) == 3 and all(lv in lw and lv in lh for lv in C3_LEVELS):
        c3 = all(lw[lv] >= lh[lv] - C3_MARGIN for lv in C3_LEVELS)
    yn = lambda b: "pending" if b is None else ("yes" if b else "no")
    if c1 is False:
        verdict = "C1 fails: the seed-1 FIVES gain was not reliable; the Weber line stops"
    elif None in (c1, c2, c3):
        verdict = "PENDING: not all runs have finished"
    elif c1 and c2 and c3:
        verdict = "CONFIRMED: at least as good as rgb on LODO and FIVES, and robust; final model"
    elif c1 and c3:
        verdict = "TRADE-OFF CONFIRMED: external gain at a LODO cost; one follow-up allowed (eye-size-relative scales)"
    else:
        verdict = "C1 holds but robustness (C3) does not: report as is; no follow-up under the rule"
    L += ["", "## Decision (rule in PROTOCOL.md, stage 6b)", "",
          f"- C1, FIVES ≥ rgb mean and above rgb in ≥ 2 seeds: **{yn(c1)}**",
          f"- C2, LODO ≥ rgb mean: **{yn(c2)}**",
          f"- C3, robustness within {C3_MARGIN} of hybrid: **{yn(c3)}**", "", f"**{verdict}**"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
