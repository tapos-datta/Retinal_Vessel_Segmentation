"""Stage 6 decision table: rules R1, R2 and R3 from PROTOCOL.md for each ablation variant, seed 1 against hybrid seed 1.

  R1  seed-1 LODO mean Dice (mean of the 4 held-out dataset means, 8-fold TTA) >= hybrid's - 0.005
  R2  Dice lost from the variant's own original images, degradation test (no TTA), at disc_32 and fives,
      at most hybrid's loss + 0.010 on the same images (results/degradation_stage6 + results/degradation)
  R3  seed-1 FIVES Dice (8-fold TTA; the 2 blank FIVES labels excluded) >= hybrid's + 0.010 (exploratory)

Usage: python stage6_summary.py [--out results/stage6]
Prints the table and writes <out>/summary.md. Missing results are shown as "–" and the verdict as pending.
"""
import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

VARIANTS = ["hybrid_masknorm", "hybrid_rg", "hybrid_floor", "hybrid_weber"]
R1_MARGIN, R2_MARGIN, R3_MARGIN = 0.005, 0.010, 0.010
R2_LEVELS = ("disc_32", "fives")


def lodo_mean(runs, variant):
    files = sorted(Path(runs).glob(f"{variant}/lodo/*/seed1/test/per_image.csv"))
    if len(files) != 4:
        return None
    return st.mean(st.mean(float(r["dice"]) for r in csv.DictReader(open(f))) for f in files)


def fives_mean(runs, variant):
    f = Path(runs) / variant / "all" / "all" / "seed1" / "external_FIVES" / "per_image.csv"
    if not f.exists():
        return None
    rows = [r for r in csv.DictReader(open(f)) if r["auc"] != "nan"]  # blank labels cannot be scored
    return st.mean(float(r["dice"]) for r in rows)


def degradation_losses(paths):
    """{variant: {level: mean Dice over seed-1 images - own original}} from one or more per_image.csv files."""
    d = defaultdict(list)
    for p in paths:
        if Path(p).exists():
            for r in csv.DictReader(open(p)):
                if r["seed"] == "seed1":
                    d[(r["variant"], r["level"])].append(float(r["dice"]))
    out = {}
    for (v, lv), xs in d.items():
        if lv != "original" and (v, "original") in d and len(xs) == len(d[(v, "original")]):
            out.setdefault(v, {})[lv] = st.mean(xs) - st.mean(d[(v, "original")])
    return out


def fmt(x):
    return "–" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--degradation", default="results/degradation_stage6/per_image.csv")
    ap.add_argument("--reference", default="results/degradation/per_image.csv")
    ap.add_argument("--out", default="results/stage6")
    args = ap.parse_args()

    h_lodo, h_fives = lodo_mean(args.runs, "hybrid"), fives_mean(args.runs, "hybrid")
    loss = degradation_losses([args.degradation, args.reference])
    h_loss = loss.get("hybrid", {})
    L = ["# Stage 6: ablation of EnhancementV3's changes on hybrid (seed 1)", "",
         f"Reference `hybrid` seed 1: LODO {fmt(h_lodo)}, FIVES {fmt(h_fives)}, degradation loss "
         + ", ".join(f"{lv} {h_loss[lv]:+.4f}" for lv in R2_LEVELS if lv in h_loss) + ".", "",
         f"R1: LODO ≥ hybrid − {R1_MARGIN}. R2: loss at {' and '.join(R2_LEVELS)} ≥ hybrid's − {R2_MARGIN}. "
         f"R3 (exploratory): FIVES ≥ hybrid + {R3_MARGIN}.", "",
         "| Variant | LODO | − hybrid | R1 | " + " | ".join(f"{lv} loss" for lv in R2_LEVELS)
         + " | R2 | FIVES | − hybrid | R3 | Reading |", "|---|---|---|---|" + "---|" * len(R2_LEVELS) + "---|---|---|---|---|"]
    for v in VARIANTS:
        lo, fv, ls = lodo_mean(args.runs, v), fives_mean(args.runs, v), loss.get(v, {})
        r1 = None if lo is None or h_lodo is None else lo >= h_lodo - R1_MARGIN
        have = all(lv in ls and lv in h_loss for lv in R2_LEVELS)
        r2 = all(ls[lv] >= h_loss[lv] - R2_MARGIN for lv in R2_LEVELS) if have else None
        r3 = None if fv is None or h_fives is None else fv >= h_fives + R3_MARGIN
        yn = lambda b: "–" if b is None else ("yes" if b else "no")
        if None in (r1, r2, r3):
            reading = "pending"
        elif r1 and r2 and r3:
            reading = "candidate fix: confirm with 3 seeds"
        elif r1 and r2:
            reading = "harmless, no FIVES effect"
        elif r3 and not r2:
            reading = "helps FIVES at the cost of robustness"
        elif r2 and r3:
            reading = "helps FIVES and keeps robustness, but hurts LODO"
        else:
            reading = "rejected"
        L.append(f"| `{v}` | {fmt(lo)} | {'–' if lo is None or h_lodo is None else f'{lo - h_lodo:+.4f}'} | {yn(r1)} | "
                 + " | ".join("–" if lv not in ls else f"{ls[lv]:+.4f}" for lv in R2_LEVELS) + f" | {yn(r2)} | "
                 f"{fmt(fv)} | {'–' if fv is None or h_fives is None else f'{fv - h_fives:+.4f}'} | {yn(r3)} | {reading} |")
    L += ["", "Seed 1 is screening only: no variant is reported as better or worse than hybrid from these runs alone."]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
