"""Exploratory: does setting every FIVES image to the same brightness remove the hybrid - rgb gap?

A 12-image check on the darkest FIVES images found that brightening them left hybrid unchanged but pulled rgb down to
hybrid's level. So rgb may score well on dark images because they are dark, not hybrid badly. This script tests that
on a brightness-stratified sample: each image is scored as it is and with a uniform gain to the same mean green
(46, the FIVES median, so bright images are dimmed and dark ones brightened). Nothing is trained.

It also records Dice at lower thresholds and AUC, because the suspected route is an operating-point shift: all models
under-segment FIVES, and a darker input may push rgb to mark more pixels as vessel. TTA is off, so baselines differ
slightly from the stage 3 numbers. FIVES has already been used for stage 3 and for choosing this question, so nothing
here is a confirmatory result, and no fix chosen from it can be claimed on FIVES.

Usage: python fives_brightness_test.py [--per-quartile 50] [--seeds 1 2 3] [--variants hybrid rgb clahe]
Resumable: finished rows in results/fives_brightness/per_image.csv are skipped.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from u2nete.data import iter_samples, load_metadata
from u2nete.infer import predict_sample
from u2nete.metrics import compute
from u2nete.model import build_model
from u2nete.utils import ROOT, configure_backend, get_device, load_json

LEVEL = 46.0
THRESHOLDS = (0.2, 0.3, 0.4)
FIELDS = ["variant", "seed", "cond", "id", "condition", "green", "gain", "quartile", "dice", "auc", "precision",
          "sensitivity"] + [f"dice_{t}" for t in THRESHOLDS]


def sample_ids(per_quartile):
    """Same images on every machine: evenly spaced through each brightness quartile, ordered by id-independent rank."""
    rows = list(csv.DictReader(open(ROOT / "results/fives_factors/fives_measures.csv")))
    labelled = {r["id"] for r in csv.DictReader(open(ROOT / "runs/hybrid/all/all/seed1/external_FIVES/per_image.csv"))
                if r["auc"] != "nan"}
    rows = sorted((r for r in rows if r["id"] in labelled), key=lambda r: float(r["brightness"]))
    out = {}
    for q in range(4):
        part = rows[q * len(rows) // 4: (q + 1) * len(rows) // 4]
        for k in np.linspace(0, len(part) - 1, min(per_quartile, len(part))).astype(int):
            out[part[k]["id"]] = q
    return out


def dice_at(prob, gt, fov, t):
    inside = fov > 0
    pred, truth = prob[inside] >= t, gt[inside] > 0
    return 2 * (pred & truth).sum() / max(pred.sum() + truth.sum(), 1)


def run(variant, seed, chosen, writer, handle, done, device):
    d = ROOT / f"runs/{variant}/all/all/seed{seed}"
    cfg = load_json(d / "config.json")
    root = ROOT / "data/external_FIVES"
    meta = load_metadata(root)
    todo = [i for i in chosen if any((variant, f"seed{seed}", c, i) not in done for c in ("original", "norm46"))]
    if not todo:
        return
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(torch.load(d / "best.pt", map_location=device)["model"])
    model.eval()
    for s in iter_samples(todo, root, cfg["data"], meta):
        original = s.image
        green = float(original[..., 1][s.fov > 0].mean())
        for cond, gain in (("original", 1.0), ("norm46", LEVEL / green)):
            if (variant, f"seed{seed}", cond, s.id) in done:
                continue
            s.image = original if gain == 1.0 else np.clip(original.astype(np.float32) * gain, 0, 255).round().astype(np.uint8)
            prob = predict_sample(model, s, cfg, device, tta=False, to_original=True)
            m = compute(prob, s.gt, s.fov, 0.5)
            writer.writerow({"variant": variant, "seed": f"seed{seed}", "cond": cond, "id": s.id,
                             "condition": meta[s.id]["condition"], "green": f"{green:.2f}", "gain": f"{gain:.4f}",
                             "quartile": chosen[s.id], **{k: f"{m[k]:.5f}" for k in ("dice", "auc", "precision", "sensitivity")},
                             **{f"dice_{t}": f"{dice_at(prob, s.gt, s.fov, t):.5f}" for t in THRESHOLDS}})
            handle.flush()
        s.image = original
        print(f"  {variant:7} seed{seed} {s.id} green {green:.0f} done", flush=True)
    del model


def summarize(path, out):
    rows = list(csv.DictReader(open(path)))
    mean = defaultdict(lambda: defaultdict(list))
    info = {}
    for r in rows:
        for k in ("dice", "auc", "precision", "sensitivity", *[f"dice_{t}" for t in THRESHOLDS]):
            mean[r["variant"], r["cond"], k, r["id"]]["v"].append(float(r[k]))
        info[r["id"]] = (int(r["quartile"]), float(r["green"]), r["condition"])
    variants = sorted({r["variant"] for r in rows})
    ids = sorted(info)
    get = lambda v, c, k: np.array([np.mean(mean[v, c, k, i]["v"]) for i in ids])
    q = np.array([info[i][0] for i in ids])
    green = np.array([info[i][1] for i in ids])
    L = [f"# FIVES at equal brightness (exploratory)\n\n{len(ids)} images, {len({r['seed'] for r in rows})} seeds, no TTA, "
         f"threshold 0.5 unless stated. Quartile 1 is darkest.\n"]
    L += ["## Mean by brightness quartile\n", "| Model | Input | Measure | Q1 | Q2 | Q3 | Q4 | All |", "|---|---|---|---|---|---|---|---|"]
    for v in variants:
        for c in ("original", "norm46"):
            for k in ("dice", "sensitivity", "precision", "auc", "dice_0.3"):
                a = get(v, c, k)
                L.append(f"| {v} | {c} | {k} | " + " | ".join(f"{a[q == j].mean():.3f}" for j in range(4)) + f" | {a.mean():.3f} |")
    L += ["\n## hybrid − other\n", "| Comparison | Input | Q1 | Q2 | Q3 | Q4 | All | better / worse | p | Spearman with brightness |", "|---|---|---|---|---|---|---|---|---|---|"]
    for other in [v for v in variants if v != "hybrid"]:
        for c in ("original", "norm46"):
            for k in ("dice", "dice_0.3", "auc"):
                g = get("hybrid", c, k) - get(other, c, k)
                L.append(f"| hybrid − {other} ({k}) | {c} | " + " | ".join(f"{g[q == j].mean():+.3f}" for j in range(4)) +
                         f" | {g.mean():+.3f} | {(g > 0).sum()} / {(g < 0).sum()} | {wilcoxon(g).pvalue:.2g} | {spearmanr(green, g)[0]:+.2f} |")
    L += ["\n## hybrid − rgb by seed (Dice, threshold 0.5)\n", "| Seed | original | norm46 |", "|---|---|---|"]
    for seed in sorted({r["seed"] for r in rows}):
        s = lambda v, c: np.mean([float(r["dice"]) for r in rows if r["variant"] == v and r["seed"] == seed and r["cond"] == c])
        L.append(f"| {seed} | {s('hybrid', 'original') - s('rgb', 'original'):+.4f} | {s('hybrid', 'norm46') - s('rgb', 'norm46'):+.4f} |")
    text = "\n".join(L) + "\n"
    Path(out).write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-quartile", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--variants", nargs="+", default=["hybrid", "rgb", "clahe"])
    ap.add_argument("--out", default="results/fives_brightness")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "per_image.csv"
    if not args.summary_only:
        device = get_device(args.device)
        configure_backend(device, memory_fraction=0.4)  # same share of the shared GPU as training
        chosen = sample_ids(args.per_quartile)
        done = set()
        if path.exists():
            done = {(r["variant"], r["seed"], r["cond"], r["id"]) for r in csv.DictReader(open(path))}
        new = not path.exists()
        with open(path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            for v in args.variants:
                for seed in args.seeds:
                    run(v, seed, chosen, writer, handle, done, device)
    summarize(path, out / "summary.md")


if __name__ == "__main__":
    main()
