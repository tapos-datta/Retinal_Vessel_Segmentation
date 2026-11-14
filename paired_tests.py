"""Paired per-image Wilcoxon tests for the final comparison (rule: PROTOCOL.md, "Paired per-image tests for the final
comparison", fixed on 2026-11-13 before computing).

Inputs are the existing evaluate.py per_image.csv files; no model is run. Blank-label images (auc = nan) are left out,
as in aggregate.py (FIVES n = 792; LODO n = 113). For each comparison, the test is run in each of seeds 1-3 and once
on the per-image mean over the three seeds.

Per test: n, median difference (A - B) with a 95% distribution-free CI (binomial order statistics), mean difference,
fraction of images favouring A / B / tied, two-sided Wilcoxon signed-rank p (scipy, zero_method="wilcox").

Output: results/paired_tests/summary.md, results/paired_tests/results.json

Usage:
    python paired_tests.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, wilcoxon

from prediction_figure import per_image_dice

SEEDS = [1, 2, 3]
DATASETS = ["CHASE_DB1", "DRIVE", "HRF", "STARE"]
ALPHA = 0.005
N_TESTS = 16
STRICT = 0.05 / N_TESTS  # 0.003125
EXPECTED_N = {"FIVES": 792, "LODO": 113}
COMPARISONS = [  # (setting, A, B)
    ("FIVES", "hybrid_weber", "rgb"),
    ("FIVES", "hybrid", "rgb"),
    ("LODO", "hybrid_weber", "rgb"),
    ("LODO", "hybrid_weber", "hybrid"),
]


def load(runs, setting, variant, seed):
    """{image id: Dice} for one variant and seed; LODO pools the four held-out test sets."""
    if setting == "FIVES":
        return per_image_dice(runs / variant / "all" / "all" / f"seed{seed}" / "external_FIVES" / "per_image.csv")
    out = {}
    for d in DATASETS:
        part = per_image_dice(runs / variant / "lodo" / f"lodo_{d}" / f"seed{seed}" / "test" / "per_image.csv")
        if set(part) & set(out):
            raise SystemExit(f"{variant} seed {seed}: image ids repeat across LODO folds")
        out.update(part)
    return out


def median_ci(x, level=0.95):
    """Distribution-free CI for the median: [x_(l), x_(n-l+1)] with the largest l such that P(B <= l-1) <= (1-level)/2,
    B ~ Binomial(n, 1/2). Returns (lo, hi, actual coverage)."""
    s, n = np.sort(x), len(x)
    a = (1 - level) / 2
    l = 0
    while binom.cdf(l, n, 0.5) <= a:  # l = number of order statistics cut from each end
        l += 1
    if l == 0:
        return float("nan"), float("nan"), float("nan")
    cover = 1 - 2 * binom.cdf(l - 1, n, 0.5)
    return float(s[l - 1]), float(s[n - l]), float(cover)


def test(a, b):
    ids = sorted(a)
    if set(a) != set(b):
        raise SystemExit("the two models were not scored on the same images")
    d = np.array([a[i] - b[i] for i in ids])
    lo, hi, cover = median_ci(d)
    p = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
    return {"n": len(d), "median": float(np.median(d)), "ci_lo": lo, "ci_hi": hi, "ci_coverage": cover,
            "mean": float(d.mean()), "frac_a": float((d > 0).mean()), "frac_b": float((d < 0).mean()),
            "frac_tie": float((d == 0).mean()), "p": p}


def mark(p):
    if p < STRICT:
        return "yes"
    if p < ALPHA:
        return "yes (between 0.0031 and 0.005)"
    return "no"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/paired_tests")
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for setting, A, B in COMPARISONS:
        da = {s: load(runs, setting, A, s) for s in SEEDS}
        db = {s: load(runs, setting, B, s) for s in SEEDS}
        for s in SEEDS:
            for name, dd in ((A, da[s]), (B, db[s])):
                if len(dd) != EXPECTED_N[setting]:
                    raise SystemExit(f"{setting} {name} seed {s}: {len(dd)} images, expected {EXPECTED_N[setting]}")
        ids = sorted(da[1])
        for s in SEEDS:
            if set(da[s]) != set(ids) or set(db[s]) != set(ids):
                raise SystemExit(f"{setting}: image sets differ between seeds")
        analyses = [(f"seed {s}", da[s], db[s]) for s in SEEDS]
        analyses.append(("3-seed per-image mean", {i: float(np.mean([da[s][i] for s in SEEDS])) for i in ids},
                         {i: float(np.mean([db[s][i] for s in SEEDS])) for i in ids}))
        for label, a, b in analyses:
            r = {"setting": setting, "a": A, "b": B, "analysis": label, **test(a, b)}
            r["significant"] = r["p"] < ALPHA
            results.append(r)
    assert len(results) == N_TESTS

    L = ["# Paired per-image tests (rule: PROTOCOL.md, fixed 2026-11-13 before computing)", "",
         "Two-sided Wilcoxon signed-rank on per-image Dice, pairs = images, from the existing evaluate.py "
         "per_image.csv files (8-fold TTA, threshold 0.5, Dice inside the FOV at original resolution); no model was "
         "run. FIVES: 792 images with a non-blank label. LODO: the 113 pooled images, each scored by the model of the "
         "fold that held its dataset out. Difference = first model − second. Median CI: 95% distribution-free "
         "(binomial order statistics; actual coverage in results.json). Significance level 0.005 (stage 2's "
         f"Bonferroni level); {N_TESTS} tests are run here, and a strict Bonferroni level for {N_TESTS} would be "
         f"{STRICT:.4f}, so p between the two is marked. Per-seed and 3-seed-mean analyses were both declared in "
         "advance; neither is the primary one.", ""]
    for setting, A, B in COMPARISONS:
        L += [f"## {setting}: `{A}` vs `{B}`", "",
              f"| analysis | n | median diff [95% CI] | mean diff | images favouring `{A}` / `{B}` / tied | "
              "Wilcoxon p | p < 0.005 |", "|---|---|---|---|---|---|---|"]
        for r in results:
            if (r["setting"], r["a"], r["b"]) != (setting, A, B):
                continue
            L.append(f"| {r['analysis']} | {r['n']} | {r['median']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
                     f"{r['mean']:+.4f} | {r['frac_a']:.1%} / {r['frac_b']:.1%} / {r['frac_tie']:.1%} | "
                     f"{r['p']:.2g} | {mark(r['p'])} |")
        per_seed = [r for r in results if (r["setting"], r["a"], r["b"]) == (setting, A, B)
                    and r["analysis"].startswith("seed")]
        signs = ["+" if r["median"] > 0 else "−" if r["median"] < 0 else "0" for r in per_seed]
        sig = [r for r in per_seed if r["significant"]]
        L += ["", f"Sign of the median difference in seeds 1 / 2 / 3: {' / '.join(signs)}"
              + (" (the same in all three seeds)." if len(set(signs)) == 1 else
                 " (not the same in all three seeds)."
                 + (" Seeds with p < 0.005 point in opposite directions."
                    if len({r['median'] > 0 for r in sig}) > 1 else "")), ""]
    L += ["## What these tests do and do not show", "",
          "- An image-level paired test treats each trained model as fixed. Its p-value measures whether the two "
          "given models differ on these images; it says nothing about training (seed) variability, which is what "
          "the seed spread in the headline chart (figures/report/results_lodo_vs_fives.png) shows. The per-seed rows "
          "show whether the result repeats across seeds; three seeds are too few to test seed variability formally.",
          "- With n = 792 on FIVES, a very small difference reaches any significance level. Read the size of the "
          "difference (median and its interval, the fraction of images favouring each model) before p.",
          "- The 3-seed per-image mean reduces seed noise in each image's Dice, but it is still a test over images, "
          "not over training runs.",
          "- FIVES informed the choice of `hybrid_weber` (stage 6), so its FIVES comparisons are not independent "
          "evidence (PROTOCOL.md, stage 6b)."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    with open(out / "results.json", "w") as fh:
        json.dump({"alpha": ALPHA, "n_tests": N_TESTS, "strict_bonferroni": STRICT, "tests": results}, fh, indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
