"""How long did each run train, and was 12,000 iterations enough?

For every finished run it reads val_log.csv (validation Dice every 1,000 iterations) and reports:
  stopped    iteration where training ended (12000 = ran the full schedule; less = early stop)
  best_it    iteration of the checkpoint that was kept (best validation Dice)
  late_best  the best checkpoint came in the last 2,000 iterations of a full-length run: the model may still
             have been improving when the schedule ended, so a longer schedule might help that run

Usage:
    python training_report.py                    # all runs under runs/
    python training_report.py --csv results/training_report.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from u2nete.utils import load_json


def read_run(d: Path):
    log = d / "val_log.csv"
    if not log.exists():
        return None
    with open(log) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    total = int(load_json(d / "config.json")["train"]["iterations"]) if (d / "config.json").exists() else 12000
    patience = int(load_json(d / "config.json")["train"]["patience"]) if (d / "config.json").exists() else 6
    finished = (d / "train_summary.json").exists()
    its = [int(r["iteration"]) for r in rows]
    vals = [float(r["val_dice"]) for r in rows]
    best_i = max(range(len(vals)), key=lambda i: (vals[i], -i))  # first occurrence of the best value
    stopped = its[-1]
    early = finished and stopped < total
    return {
        "variant": d.parts[-4], "split": d.parts[-3], "fold": d.parts[-2], "seed": d.parts[-1],
        "finished": finished, "stopped": stopped, "total": total, "early_stop": early,
        "best_it": its[best_i], "best_val_dice": vals[best_i], "patience": patience,
        "late_best": finished and not early and its[best_i] > total - 2000,
        # gain of the final 4 validations over the best before them: how much the last third still added
        "gain_last_third": round(max(vals) - max(vals[: max(1, len(vals) - 4)]), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--csv", help="also write one row per run to this file")
    args = ap.parse_args()

    runs = [r for d in sorted(Path(args.runs).glob("*/*/*/seed*")) if (r := read_run(d))]
    if not runs:
        raise SystemExit(f"no runs with val_log.csv under {args.runs}/")
    done = [r for r in runs if r["finished"]]
    print(f"{len(runs)} runs found, {len(done)} finished, {len(runs) - len(done)} still running or interrupted\n")

    groups = defaultdict(list)
    for r in done:
        groups[(r["variant"], r["split"])].append(r)
    print(f"{'variant':30} {'split':7} {'runs':>4} {'early stop':>10} {'full 12k':>8} {'best it (median, range)':>24} "
          f"{'late best':>9}")
    for (v, s), rs in sorted(groups.items()):
        best = sorted(r["best_it"] for r in rs)
        med = best[len(best) // 2]
        print(f"{v:30} {s:7} {len(rs):4} {sum(r['early_stop'] for r in rs):10} {sum(not r['early_stop'] for r in rs):8} "
              f"{f'{med} ({best[0]}-{best[-1]})':>24} {sum(r['late_best'] for r in rs):9}")

    n = len(done)
    if not n:
        return
    early = sum(r["early_stop"] for r in done)
    late = sum(r["late_best"] for r in done)
    stops = sorted(r["stopped"] for r in done if r["early_stop"])
    print(f"\nAll finished runs: {n}")
    print(f"  stopped early (no validation gain for {done[0]['patience']} checks): {early} ({100 * early / n:.0f}%)"
          + (f", median stop at iteration {stops[len(stops) // 2]}" if stops else ""))
    print(f"  ran the full schedule: {n - early}")
    print(f"  best checkpoint in the last 2,000 iterations of a full run: {late} ({100 * late / n:.0f}%)")
    gains = sorted(r["gain_last_third"] for r in done if not r["early_stop"])
    if gains:
        print(f"  full-length runs, validation Dice gained in the last 4,000 iterations: "
              f"median {gains[len(gains) // 2]:.4f}, max {gains[-1]:.4f}")
    print("\nReading it: many early stops and few late bests mean 12,000 iterations is enough. Many late bests with a "
          "real gain in the last third (> ~0.003) mean training was cut short; then try train.iterations=20000.")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(runs[0]))
            w.writeheader()
            w.writerows(runs)
        print(f"\nper-run table: {args.csv}")


if __name__ == "__main__":
    main()
