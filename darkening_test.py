"""Does the enhancement module fail because images are dark? A controlled test that does not touch FIVES.

On FIVES, hybrid's deficit against rgb grew as images got darker, but that link was found after seeing the results,
and dark FIVES images may differ in other ways too (camera, blur, disease). This test changes brightness alone:
each trained LODO model (no new training) sees its own held-out test images, darkened step by step to set brightness
levels, and every variant sees exactly the same images at each level.

Brightness = mean green value inside the FOV, the measure used for FIVES. Levels: 46 and 38 are FIVES quartile
boundaries; 25 and 15 reach its darkest images. Images are rounded to 8 bits as a real dark photograph would be, and
never brightened: images already darker than a level stay as dark as they are ("darkened" counts those that changed).
Two kinds of darkening:
  gain    one multiplier for the whole image (pure under-exposure)
  uneven  a smooth brightness ramp in a direction fixed per image, with the same mean. Its p95/p5 range is 3, the
          median shading measured on the darkest quarter of FIVES (2.97; CHASE_DB1 2.8, HRF 6.1, DRIVE 1.4, STARE 1.9).

Decision for each kind, same rule:
  SUPPORTED     hybrid - rgb falls at every step as brightness falls AND is negative at level 38 or darker, all 3 seeds
  NOT SUPPORTED otherwise: this kind of darkness does not reproduce the FIVES pattern
The rule was written before any results. The uneven arm was added after an 8-image trial of the gain arm, in which
hybrid hardly changed: its per-patch min-max cancels a uniform gain by construction, so gain alone is a weak test of it.
Limit: neither kind imitates the blur, colour casts or reduced vessel contrast of real dark images.

Usage (safe to stop and restart; finished predictions are kept):
    python darkening_test.py --no-tta                 # 3 variants x 3 seeds x 4 folds, roughly 2 h on an RTX 4090
    python darkening_test.py                          # with 8-fold TTA, roughly 8x longer; baseline matches reported Dice
    python darkening_test.py --summary-only           # rebuild the summary from what has finished
Writes results/darkening/per_image.csv and summary.md.

A new variant is tested in its own folder, against the frozen hybrid/rgb/clahe rows of the original test, which were
made without TTA (so use --no-tta). Only reference rows with the same images, seeds and levels are compared:
    python darkening_test.py --variants enh3 --seeds 1 --no-tta --out results/darkening_enh3 \\
        --reference results/darkening/per_image.csv
The SUPPORTED rule above was written for hybrid and is not applied to these runs; they report the differences only.
"""
import argparse
import csv
import zlib
from collections import defaultdict
from pathlib import Path

from u2nete.utils import ROOT, configure_backend, get_device, load_json

import numpy as np
import torch
from scipy.stats import wilcoxon

from u2nete.data import iter_samples, load_metadata
from u2nete.infer import predict_sample
from u2nete.metrics import METRIC_KEYS, compute
from u2nete.model import build_model

ORIGINAL = "original"
CONTROLS = ("hybrid", "rgb", "clahe")


def green_mean(image, fov):
    return float(image[..., 1][fov > 0].mean())


SHADING_RANGE = 3.0  # p95/p5 of the uneven field; see the docstring


def uneven_field(fov, image_id):
    """Smooth log-linear ramp with p95/p5 = SHADING_RANGE inside the FOV; direction depends only on the image id."""
    h, w = fov.shape
    angle = (zlib.crc32(image_id.encode()) % 3600) / 3600 * 2 * np.pi
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = xs * np.cos(angle) + ys * np.sin(angle)
    lo, hi = np.percentile(t[fov > 0], [5, 95])
    return np.exp(np.log(SHADING_RANGE) * (t - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def darken(image, fov, level, image_id=""):
    """Returns (darkened image, mean gain inside the FOV). Darken only: the gain never exceeds 1 at any pixel."""
    if level == ORIGINAL:
        return image, 1.0
    arm, value = level.split("_")
    target, green = float(value), image[..., 1].astype(np.float32)
    inside = fov > 0
    if arm == "gain":
        gain = min(1.0, target / max(float(green[inside].mean()), 1e-6))
        if gain >= 1.0:
            return image, 1.0
        field = np.full(fov.shape, gain, np.float32)
    elif arm == "uneven":
        base = uneven_field(fov, image_id)
        base /= base[inside].max()
        scale = 1.0
        for _ in range(8):  # brighten the field towards the target; capped pixels need a few rounds to converge
            mean = float((green * np.minimum(base * scale, 1.0))[inside].mean())
            scale *= target / max(mean, 1e-6)
        field = np.minimum(base * scale, 1.0)
    else:
        raise ValueError(level)
    out = np.clip(np.round(image.astype(np.float32) * field[..., None]), 0, 255).astype(np.uint8)
    return out, float(field[inside].mean())


def run_model(run_dir, levels, tta, done, writer, handle, device, limit):
    cfg = load_json(run_dir / "config.json")
    pool = Path(cfg["data"]["pool"])
    pool = pool if pool.is_absolute() else ROOT / pool
    meta = load_metadata(pool)
    ids = load_json(run_dir / "fold.json")["test"][:limit]
    variant, split, fold, seed = run_dir.parts[-4:]
    todo = [i for i in ids if any((variant, fold, seed, str(lv), i) not in done for lv in levels)]
    if not todo:
        return
    ck = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    for s in iter_samples(todo, pool, cfg["data"], meta):
        before = green_mean(s.image, s.fov)
        original = s.image
        for lv in levels:
            if (variant, fold, seed, str(lv), s.id) in done:
                continue
            s.image, gain = darken(original, s.fov, lv, s.id)
            prob = predict_sample(model, s, cfg, device, tta=tta, to_original=True)
            m = compute(prob, s.gt, s.fov, cfg["inference"]["threshold"])
            writer.writerow({"variant": variant, "fold": fold, "seed": seed, "level": lv, "id": s.id,
                             "dataset": s.dataset, "green_before": f"{before:.2f}", "gain": f"{gain:.4f}",
                             **{k: f"{m[k]:.5f}" for k in METRIC_KEYS}})
            handle.flush()
        s.image = original
        print(f"  {variant:7} {fold:16} {seed} {s.id:14} green {before:5.1f} done", flush=True)
    del model


def check_baseline(rows, runs_root):
    """With TTA, the unchanged images must reproduce the Dice values already reported for these runs."""
    worst, n = 0.0, 0
    for r in rows:
        if r["level"] != ORIGINAL:
            continue
        f = Path(runs_root) / r["variant"] / "lodo" / r["fold"] / r["seed"] / "test" / "per_image.csv"
        if not f.exists():
            continue
        if not hasattr(check_baseline, "cache"):
            check_baseline.cache = {}
        if f not in check_baseline.cache:
            with open(f) as fh:
                check_baseline.cache[f] = {x["id"]: float(x["dice"]) for x in csv.DictReader(fh)}
        worst = max(worst, abs(check_baseline.cache[f][r["id"]] - float(r["dice"])))
        n += 1
    return worst, n


def merge_reference(rows, reference):
    """Add reference rows for the same images, seeds and levels as `rows`, so every comparison is like for like."""
    with open(reference) as f:
        ref = list(csv.DictReader(f))
    tested, seeds, ids, lvls = (({r[k] for r in rows}) for k in ("variant", "seed", "id", "level"))
    return rows + [r for r in ref if r["variant"] not in tested and r["seed"] in seeds
                   and r["id"] in ids and r["level"] in lvls]


def summarize(path, out, levels, tta, runs_root, reference=None):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("no results yet")
    if reference:
        rows = merge_reference(rows, reference)
    keys = [k for k in levels if any(r["level"] == k for r in rows)]
    present = {r["variant"] for r in rows}
    new_variants = sorted(present - set(CONTROLS))
    variants = [v for v in CONTROLS if v in present] + new_variants
    seeds = sorted({r["seed"] for r in rows})

    img = defaultdict(list)  # (variant, level, id) -> Dice per seed
    by_seed = defaultdict(list)  # (variant, level, seed, id) -> Dice
    ds_of, gain_of = {}, {}
    for r in rows:
        img[(r["variant"], r["level"], r["id"])].append(float(r["dice"]))
        by_seed[(r["variant"], r["level"], r["seed"])].append((r["id"], float(r["dice"])))
        ds_of[r["id"]] = r["dataset"]
        gain_of[(r["level"], r["id"])] = float(r["gain"])

    def paired(a, b, lv, ids=None):
        common = [k[2] for k in img if k[0] == a and k[1] == lv and (b, lv, k[2]) in img]
        if ids is not None:
            common = [i for i in common if i in ids]
        x = np.array([np.mean(img[(a, lv, i)]) - np.mean(img[(b, lv, i)]) for i in common])
        return common, x

    L = ["# Darkening test", "",
         f"LODO models on their own held-out images, {'8-fold TTA' if tta else 'no TTA'}, threshold 0.5. "
         f"Variants: {', '.join(variants)}; seeds: {len(seeds)}. Level = target mean green inside the FOV.", ""]
    if reference:
        L += [f"Control rows come from `{reference}`, limited to the same images, seeds and levels.", ""]
    if tta and ORIGINAL in keys:
        worst, n = check_baseline(rows, runs_root)
        if n:
            L += [f"Baseline check: {n} unchanged predictions differ from the reported per-image Dice by at most {worst:.5f}.", ""]

    L += ["## Mean Dice by brightness level", "",
          "| Level | darkened | " + " | ".join(variants) + " | hybrid − rgb | better / worse | p | hybrid − clahe | better / worse | p |",
          "|---|---|" + "---|" * len(variants) + "---|---|---|---|---|---|"]
    ids_all = sorted({r["id"] for r in rows})
    for k in keys:
        cells = [f"{np.mean([np.mean(img[(v, k, i)]) for i in ids_all if (v, k, i) in img]):.4f}" for v in variants]
        n_dark = sum(gain_of.get((k, i), 1.0) < 1.0 for i in ids_all)
        extra = []
        for other in ("rgb", "clahe"):
            if "hybrid" in variants and other in variants:
                _, x = paired("hybrid", other, k)
                p = wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else float("nan")
                extra += [f"{x.mean():+.4f}", f"{(x > 0).sum()} / {(x < 0).sum()}", f"{p:.2g}"]
            else:
                extra += ["–", "–", "–"]
        L.append(f"| {k} | {n_dark} | " + " | ".join(cells) + " | " + " | ".join(extra) + " |")

    if "hybrid" in variants and "rgb" in variants:
        L += ["", "## hybrid − rgb, by held-out dataset", "",
              "| Level | " + " | ".join(sorted(set(ds_of.values()))) + " |", "|---|" + "---|" * len(set(ds_of.values()))]
        for k in keys:
            cells = []
            for d in sorted(set(ds_of.values())):
                _, x = paired("hybrid", "rgb", k, {i for i, v in ds_of.items() if v == d})
                cells.append(f"{x.mean():+.4f} ({(x > 0).sum()}/{(x < 0).sum()})" if len(x) else "–")
            L.append(f"| {k} | " + " | ".join(cells) + " |")

        L += ["", "## hybrid − rgb, by seed (same-seed means)", "", "| Level | " + " | ".join(seeds) + " |",
              "|---|" + "---|" * len(seeds)]
        per_level = {}
        for k in keys:
            vals = []
            for s in seeds:
                h, r = dict(by_seed[("hybrid", k, s)]), dict(by_seed[("rgb", k, s)])
                common = sorted(set(h) & set(r))
                vals.append(float(np.mean([h[i] - r[i] for i in common])) if common else float("nan"))
            per_level[k] = vals
            L.append(f"| {k} | " + " | ".join(f"{v:+.4f}" for v in vals) + " |")

    datasets = sorted(set(ds_of.values()))
    for v in new_variants:
        for ref in CONTROLS:
            if ref not in variants:
                continue
            L += ["", f"## {v} − {ref}", "",
                  f"| Level | {v} | {ref} | difference | better / worse | p | " + " | ".join(datasets) + " |",
                  "|---|---|---|---|---|---|" + "---|" * len(datasets)]
            for k in keys:
                common, x = paired(v, ref, k)
                if not common:
                    continue
                p = wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else float("nan")
                by_ds = []
                for d in datasets:
                    _, y = paired(v, ref, k, {i for i in common if ds_of[i] == d})
                    by_ds.append(f"{y.mean():+.4f}" if len(y) else "–")
                a, b = (np.mean([np.mean(img[(w, k, i)]) for i in common]) for w in (v, ref))
                L.append(f"| {k} | {a:.4f} | {b:.4f} | {x.mean():+.4f} | {(x > 0).sum()} / {(x < 0).sum()} | {p:.2g} | "
                         + " | ".join(by_ds) + " |")

    if "hybrid" in variants and "rgb" in variants and not new_variants:
        complete = len(keys) == len(levels) and len(seeds) == 3 and len(ids_all) == 113
        L += ["", "## Decision", ""]
        for arm in ("gain", "uneven"):
            order = [ORIGINAL] + [k for k in keys if k.startswith(arm + "_")]
            if len(order) < 2 or ORIGINAL not in keys:
                continue
            means = [float(np.mean(per_level[k])) for k in order]
            steady = all(b <= a + 1e-9 for a, b in zip(means, means[1:]))
            dark = [k for k in order[1:] if float(k.split("_")[1]) <= 38]
            negative = bool(dark) and all(all(v < 0 for v in per_level[k]) for k in dark)
            verdict = "SUPPORTED" if steady and negative else "NOT SUPPORTED"
            L += [f"**{arm}**: hybrid − rgb {', '.join(f'{m:+.4f}' for m in means)}; falls at every step: "
                  f"{'yes' if steady else 'no'}; negative in every seed at level 38 and darker: "
                  f"{'yes' if negative else 'no'}. **Rule: {verdict}**"
                  + ("" if complete else " (INCOMPLETE: not all runs have finished)"), ""]
    L += ["", "Paired tests use per-image Dice averaged over seeds; they treat images, not seeds, as the sample. "
          "Neither kind of darkening imitates blur, colour casts or reduced vessel contrast."]
    Path(out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--variants", nargs="*", default=["hybrid", "rgb", "clahe"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--levels", nargs="*", type=int, default=[46, 38, 25, 15])
    ap.add_argument("--arms", nargs="*", choices=["gain", "uneven"], default=["gain", "uneven"])
    ap.add_argument("--out", default="results/darkening")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="images per fold (quick test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--reference", help="per_image.csv of an earlier run, for the control variants (see the top of this file)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "per_image.csv"
    levels = [ORIGINAL] + [f"{arm}_{v}" for arm in args.arms for v in sorted(args.levels, reverse=True)]
    tta = not args.no_tta

    if not args.summary_only:
        dirs = [d for v in args.variants for s in args.seeds
                for d in sorted(Path(args.runs).glob(f"{v}/lodo/*/seed{s}")) if (d / "best.pt").exists()]
        if not dirs:
            raise SystemExit(f"no LODO checkpoints found under {args.runs}/")
        done = set()
        if path.exists():
            with open(path) as f:
                done = {(r["variant"], r["fold"], r["seed"], r["level"], r["id"]) for r in csv.DictReader(f)}
        device = get_device(args.device)
        cfg = load_json(dirs[0] / "config.json")
        configure_backend(device, cfg["train"].get("gpu_memory_fraction"))  # the GPU may be shared
        print(f"{len(dirs)} models, levels {levels}, TTA {'on' if tta else 'off'}, device {device}", flush=True)
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["variant", "fold", "seed", "level", "id", "dataset", "green_before",
                                              "gain", *METRIC_KEYS])
            if new:
                w.writeheader()
            for n, d in enumerate(dirs, 1):
                print(f"=== [{n}/{len(dirs)}] {d} ===", flush=True)
                run_model(d, levels, tta, done, w, f, device, args.limit)
    summarize(path, out / "summary.md", levels, tta, args.runs, args.reference)


if __name__ == "__main__":
    main()
