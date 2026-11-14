"""Can a threshold set from each image's own probability map beat the fixed 0.5 on unseen datasets?

Each trained LODO model (no new training) predicts its held-out test images, and the same probability map is turned
into a vessel map by every method below. None of the methods looks at the test labels.
  fixed_0.5    the current rule
  val_fixed    one threshold per model, the best on that model's own validation images (grid 0.10 to 0.90)
  otsu_prob    Otsu's threshold on the probabilities inside the FOV of this image
  otsu_logit   the same on log-odds, which spreads out probabilities that are piled up near 0 and 1
  hysteresis   pixels above 0.3 that connect to a pixel above 0.5

FIVES is not used here. A method is chosen on LODO and only then applied to FIVES, once.

Usage (safe to stop and restart; finished results are kept):
    python threshold_test.py                          # hybrid, rgb, clahe, enh3 on seed 1, with 8-fold TTA
    python threshold_test.py --summary-only           # rebuild the summary from what has finished
Writes results/thresholds/per_image.csv, val_thresholds.csv and summary.md.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from u2nete.utils import ROOT, configure_backend, get_device, load_json

import numpy as np
import torch
from scipy.stats import wilcoxon
from skimage.filters import apply_hysteresis_threshold, threshold_otsu

from u2nete.data import iter_samples, load_metadata
from u2nete.infer import predict_sample
from u2nete.io import read_binary
from u2nete.metrics import compute
from u2nete.model import build_model

METHODS = ["fixed_0.5", "val_fixed", "otsu_prob", "otsu_logit", "hysteresis"]
GRID = np.round(np.arange(0.10, 0.901, 0.05), 2)
FIELDS = ["variant", "fold", "seed", "id", "dataset", "method", "dice", "precision", "sensitivity"]
HYSTERESIS = (0.3, 0.5)
LOGIT_CLIP = 1e-6


def binarize(prob, fov, method, val_threshold=None):
    """Binary vessel map from a probability map. Uses the FOV but never the labels."""
    inside = fov > 0
    if method == "fixed_0.5":
        return prob >= 0.5
    if method == "val_fixed":
        return prob >= val_threshold
    if method == "otsu_prob":
        return prob >= threshold_otsu(prob[inside])
    if method == "otsu_logit":
        p = np.clip(prob, LOGIT_CLIP, 1 - LOGIT_CLIP)
        logit = np.log(p / (1 - p))
        return logit >= threshold_otsu(logit[inside])
    if method == "hysteresis":
        return apply_hysteresis_threshold(prob, *HYSTERESIS)
    raise ValueError(method)


def score(mask, gt, fov):
    return compute(mask.astype(np.float32), gt, fov, 0.5, with_cldice=False)


def predict_set(model, ids, pool, cfg, meta, device, tta):
    for s in iter_samples(ids, pool, cfg["data"], meta):
        prob = predict_sample(model, s, cfg, device, tta=tta, to_original=True)
        yield s, prob, read_binary(pool / "masks" / f"{s.id}.png"), read_binary(pool / "fov" / f"{s.id}.png")


def best_val_threshold(model, ids, pool, cfg, meta, device, tta):
    """Fixed threshold with the highest mean Dice over this model's validation images."""
    dice = np.zeros((len(ids), len(GRID)))
    for n, (_, prob, gt, fov) in enumerate(predict_set(model, ids, pool, cfg, meta, device, tta)):
        for k, t in enumerate(GRID):
            dice[n, k] = score(prob >= t, gt, fov)["dice"]
    return float(GRID[int(np.argmax(dice.mean(axis=0)))])


def run_model(run_dir, tta, done, val_thresholds, writer, handle, val_writer, val_handle, device, limit):
    fold_info = load_json(run_dir / "fold.json")
    variant, _, fold, seed = run_dir.parts[-4:]
    ids = fold_info["test"][:limit]
    todo = [i for i in ids if any((variant, fold, seed, i, m) not in done for m in METHODS)]
    if not todo:
        return
    cfg = load_json(run_dir / "config.json")
    pool = Path(cfg["data"]["pool"])
    pool = pool if pool.is_absolute() else ROOT / pool
    meta = load_metadata(pool)
    ck = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    key = (variant, fold, seed)
    if key not in val_thresholds:
        val_thresholds[key] = best_val_threshold(model, fold_info["val"][:limit], pool, cfg, meta, device, tta)
        val_writer.writerow({"variant": variant, "fold": fold, "seed": seed, "threshold": val_thresholds[key]})
        val_handle.flush()
    for s, prob, gt, fov in predict_set(model, todo, pool, cfg, meta, device, tta):
        for m in METHODS:
            if (variant, fold, seed, s.id, m) in done:
                continue
            r = score(binarize(prob, fov, m, val_thresholds[key]), gt, fov)
            writer.writerow({"variant": variant, "fold": fold, "seed": seed, "id": s.id, "dataset": s.dataset, "method": m,
                             **{k: f"{r[k]:.5f}" for k in ("dice", "precision", "sensitivity")}})
        handle.flush()
        print(f"  {variant:7} {fold:16} {seed} {s.id} done", flush=True)
    del model


def summarize(out, tta):
    with open(out / "per_image.csv") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("no results yet")
    img = defaultdict(list)
    ds_of = {}
    for r in rows:
        for k in ("dice", "precision", "sensitivity"):
            img[(r["variant"], r["method"], r["id"], k)].append(float(r[k]))
        ds_of[r["id"]] = r["dataset"]
    variants = [v for v in ("hybrid", "rgb", "clahe", "enh3") if any(r["variant"] == v for r in rows)]
    variants += sorted({r["variant"] for r in rows} - set(variants))
    datasets = sorted(set(ds_of.values()))

    def values(v, m, k, ids):
        return np.array([np.mean(img[(v, m, i, k)]) for i in ids if (v, m, i, k) in img])

    def lodo(v, m, k):
        return float(np.mean([values(v, m, k, [i for i in ds_of if ds_of[i] == d]).mean() for d in datasets]))

    L = ["# Image-adaptive thresholds", "", f"LODO models on their held-out images, {'8-fold TTA' if tta else 'no TTA'}. "
         f"Images: {len(ds_of)}. Seeds: {len({r['seed'] for r in rows})}. LODO mean = mean of the per-dataset means.", ""]
    for v in variants:
        L += [f"## {v}", "", "| Method | LODO Dice | difference from fixed_0.5 | better / worse | p | precision | sensitivity | "
              + " | ".join(datasets) + " |", "|---|---|---|---|---|---|---|" + "---|" * len(datasets)]
        base = lodo(v, "fixed_0.5", "dice")
        for m in METHODS:
            ids = sorted(ds_of)
            x = values(v, m, "dice", ids) - values(v, "fixed_0.5", "dice", ids)
            p = wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else float("nan")
            by_ds = [f"{values(v, m, 'dice', [i for i in ds_of if ds_of[i] == d]).mean():.3f}" for d in datasets]
            L.append(f"| {m} | {lodo(v, m, 'dice'):.4f} | {lodo(v, m, 'dice') - base:+.4f} | {(x > 0).sum()} / {(x < 0).sum()} | "
                     f"{p:.2g} | {lodo(v, m, 'precision'):.3f} | {lodo(v, m, 'sensitivity'):.3f} | " + " | ".join(by_ds) + " |")
        L.append("")
    L += ["Precision and sensitivity are LODO means. Paired tests use per-image Dice and treat images as the sample."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--variants", nargs="*", default=["hybrid", "rgb", "clahe", "enh3"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    ap.add_argument("--out", default="results/thresholds")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="images per fold (quick test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tta = not args.no_tta
    if not args.summary_only:
        dirs = [d for v in args.variants for s in args.seeds
                for d in sorted(Path(args.runs).glob(f"{v}/lodo/*/seed{s}")) if (d / "best.pt").exists()]
        if not dirs:
            raise SystemExit(f"no LODO checkpoints found under {args.runs}/")
        device = get_device(args.device)
        configure_backend(device, load_json(dirs[0] / "config.json")["train"].get("gpu_memory_fraction"))  # the GPU may be shared
        path, val_path = out / "per_image.csv", out / "val_thresholds.csv"
        done, val_thresholds = set(), {}
        if path.exists():
            with open(path) as f:
                done = {(r["variant"], r["fold"], r["seed"], r["id"], r["method"]) for r in csv.DictReader(f)}
        if val_path.exists():
            with open(val_path) as f:
                val_thresholds = {(r["variant"], r["fold"], r["seed"]): float(r["threshold"]) for r in csv.DictReader(f)}
        new, new_val = not path.exists(), not val_path.exists()
        print(f"{len(dirs)} models, TTA {'on' if tta else 'off'}, device {device}", flush=True)
        with open(path, "a", newline="") as f, open(val_path, "a", newline="") as vf:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            vw = csv.DictWriter(vf, fieldnames=["variant", "fold", "seed", "threshold"])
            if new:
                w.writeheader()
            if new_val:
                vw.writeheader()
            for n, d in enumerate(dirs, 1):
                print(f"=== [{n}/{len(dirs)}] {d} ===", flush=True)
                run_model(d, tta, done, val_thresholds, w, f, vw, vf, device, args.limit)
    summarize(out, tta)


if __name__ == "__main__":
    main()
