"""Is the FIVES gap about the images or about how FIVES vessels were labelled?

All models miss vessels on FIVES (precision about 0.96, sensitivity 0.62-0.66) but are balanced on LODO. If FIVES
annotators draw vessels wider than the training labels, a model can find every vessel and still lose Dice. This test
separates "finding the vessel" from "matching its drawn width", using the frozen models (no training):

  cl_recall     share of the labelled centreline (skeleton) covered by the prediction: did the model find the vessel?
  cl_precision  share of the predicted centreline lying inside a labelled vessel
  width_ratio   predicted / labelled vessel width, measured at the same labelled centreline points the model found
  sensitivity   ordinary pixel sensitivity, for comparison

Each model is measured on data it never trained on: LODO models on their held-out images, `all` models on FIVES.
Label statistics (vessel width in pixels and relative to the FOV diameter) are written for every dataset.
The decision rule is in PROTOCOL.md ("Labelling-style test"); it was fixed before any real result.

Usage (safe to stop and restart; finished images are kept):
    python label_style_test.py                     # hybrid, rgb, clahe, enh3; seeds 1-3; LODO and FIVES; no TTA
    python label_style_test.py --summary-only
Writes results/label_style/per_image.csv, label_stats.csv and summary.md.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from u2nete.utils import ROOT, configure_backend, get_device, load_json

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from u2nete.data import iter_samples, load_metadata
from u2nete.infer import predict_sample
from u2nete.io import read_binary
from u2nete.metrics import compute
from u2nete.model import build_model

VARIANTS = ["hybrid", "rgb", "clahe", "enh3", "hybrid_weber"]
FIELDS = ["source", "variant", "fold", "seed", "id", "dataset", "condition",
          "dice", "precision", "sensitivity", "cl_recall", "cl_precision", "width_ratio"]
LABEL_FIELDS = ["id", "dataset", "condition", "fov_diameter", "width_px", "width_rel", "vessel_fraction"]


def widths(mask):
    """Local vessel diameter at every pixel inside the mask: 2 x distance to the nearest background pixel - 1."""
    return 2.0 * distance_transform_edt(mask) - 1.0


def label_stats(gt, fov, fov_diameter):
    g = gt.astype(bool) & fov.astype(bool)
    sk = skeletonize(g)
    w = widths(g)[sk]
    return {"width_px": float(w.mean()), "width_rel": float(w.mean() / fov_diameter),
            "vessel_fraction": float(g.sum() / max(fov.astype(bool).sum(), 1))}


def style_metrics(pred, gt, fov):
    """Centreline recall/precision and matched width ratio. pred, gt, fov: binary arrays of the same shape."""
    f = fov.astype(bool)
    p, g = pred.astype(bool) & f, gt.astype(bool) & f
    sk_g, sk_p = skeletonize(g), skeletonize(p)
    found = sk_g & p
    wp, wg = widths(p), widths(g)
    return {"cl_recall": float(found.sum() / max(sk_g.sum(), 1)),
            "cl_precision": float((sk_p & g).sum() / max(sk_p.sum(), 1)),
            "width_ratio": float(wp[found].mean() / max(wg[found].mean(), 1e-6)) if found.any() else float("nan")}


def jobs(runs, variants, seeds, external):
    out = []
    for v in variants:
        for s in seeds:
            out += [("lodo", d) for d in sorted(Path(runs).glob(f"{v}/lodo/*/seed{s}")) if (d / "best.pt").exists()]
            d = Path(runs) / v / "all" / "all" / f"seed{s}"
            if (d / "best.pt").exists() and external:
                out.append(("fives", d))
    return out


def run_model(source, run_dir, external, tta, done, writer, handle, device, limit):
    cfg = load_json(run_dir / "config.json")
    variant, _, fold, seed = run_dir.parts[-4:]
    if source == "fives":
        pool = Path(external)
        ids = sorted(load_metadata(pool if pool.is_absolute() else ROOT / pool))
    else:
        pool = Path(cfg["data"]["pool"])
        ids = load_json(run_dir / "fold.json")["test"]
    pool = pool if pool.is_absolute() else ROOT / pool
    ids = ids[:limit]
    todo = [i for i in ids if (source, variant, fold, seed, i) not in done]
    if not todo:
        return
    meta = load_metadata(pool)
    ck = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    thr = cfg["inference"]["threshold"]
    for s in iter_samples(todo, pool, cfg["data"], meta):
        gt, fov = read_binary(pool / "masks" / f"{s.id}.png"), read_binary(pool / "fov" / f"{s.id}.png")
        if not (gt.astype(bool) & fov.astype(bool)).any():
            continue  # blank label: cannot be scored (2 FIVES images)
        prob = predict_sample(model, s, cfg, device, tta=tta, to_original=True)
        m = compute(prob, gt, fov, thr, with_cldice=False)
        st = style_metrics(prob >= thr, gt, fov)
        writer.writerow({"source": source, "variant": variant, "fold": fold, "seed": seed, "id": s.id,
                         "dataset": s.dataset, "condition": meta[s.id].get("condition", ""),
                         **{k: f"{m[k]:.5f}" for k in ("dice", "precision", "sensitivity")},
                         **{k: f"{v:.5f}" for k, v in st.items()}})
        handle.flush()
    print(f"  {source} {variant} {fold} {seed}: {len(todo)} images", flush=True)
    del model


def write_label_stats(path, pools, limit):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for pool in pools:
            meta = load_metadata(pool)
            for i in sorted(meta)[:limit]:
                gt, fov = read_binary(pool / "masks" / f"{i}.png"), read_binary(pool / "fov" / f"{i}.png")
                if not (gt.astype(bool) & fov.astype(bool)).any():
                    continue
                st = label_stats(gt, fov, float(meta[i]["fov_diameter"]))
                w.writerow({"id": i, "dataset": meta[i]["dataset"], "condition": meta[i].get("condition", ""),
                            "fov_diameter": meta[i]["fov_diameter"], **{k: f"{v:.6f}" for k, v in st.items()}})


METRICS = ["dice", "sensitivity", "precision", "cl_recall", "cl_precision", "width_ratio"]


def summarize(out):
    with open(out / "per_image.csv") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("no results yet")
    # per (source, variant, seed): LODO = mean of per-dataset means; FIVES = mean over images
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[(r["source"], r["variant"], r["seed"])][r["dataset"]].append({k: float(r[k]) for k in METRICS})

    def value(source, variant, k):
        per_seed = []
        for (src, v, s), by_ds in acc.items():
            if src == source and v == variant:
                per_seed.append(np.mean([np.nanmean([x[k] for x in xs]) for xs in by_ds.values()]))
        return float(np.mean(per_seed)) if per_seed else float("nan"), len(per_seed)

    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    L = ["# Labelling-style test", "", "Frozen models on data they never trained on, threshold 0.5. "
         "LODO = mean of the 4 held-out dataset means; FIVES = mean over its images; both averaged over seeds.", "",
         "| Variant | Seeds (LODO / FIVES) | Measure | LODO | FIVES | FIVES − LODO |", "|---|---|---|---|---|---|"]
    verdicts = {}
    for v in variants:
        diff = {}
        for k in METRICS:
            (a, na), (b, nb) = value("lodo", v, k), value("fives", v, k)
            diff[k] = b - a
            L.append(f"| {v} | {na} / {nb} | {k} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
        found = abs(diff["cl_recall"]) <= 0.05 and diff["sensitivity"] <= -0.10
        thinner = diff["width_ratio"] <= -0.15
        verdicts[v] = (found, thinner)
    L += ["", "## Decision (rule in PROTOCOL.md)", "",
          "| Variant | (a) vessels found, pixels missed | (b) predicted width thinner on FIVES |", "|---|---|---|"]
    for v, (a, b) in verdicts.items():
        L.append(f"| {v} | {'yes' if a else 'no'} | {'yes' if b else 'no'} |")
    complete = len(variants) == len(VARIANTS) and all(value(s, v, "dice")[1] == 3 for s in ("lodo", "fives") for v in variants)
    if all(a and b for a, b in verdicts.values()):
        verdict = "SUPPORTED: labelling style explains most of the FIVES gap"
    elif not any(a or b for a, b in verdicts.values()):
        verdict = "NOT SUPPORTED: the FIVES gap is not a labelling-style effect"
    else:
        verdict = "MIXED: labelling style explains part of the gap, or only for some models"
    L += ["", f"**{verdict}**" + ("" if complete else " (INCOMPLETE: not all models have finished)"), ""]

    # secondary: does hybrid's deficit survive when only finding the vessel counts?
    L += ["## Secondary: differences between models on FIVES", "", "| Comparison | Dice | cl_recall | width_ratio |",
          "|---|---|---|---|"]
    for a, b in (("hybrid", "rgb"), ("enh3", "rgb"), ("hybrid", "clahe")):
        if a in variants and b in variants:
            L.append(f"| {a} − {b} | " + " | ".join(
                f"{value('fives', a, k)[0] - value('fives', b, k)[0]:+.4f}" for k in ("dice", "cl_recall", "width_ratio")) + " |")

    lpath = out / "label_stats.csv"
    if lpath.exists():
        with open(lpath) as f:
            lab = list(csv.DictReader(f))
        L += ["", "## Label statistics by dataset", "", "| Dataset | Images | Mean width (px) | Width / FOV diameter (x1000) | "
              "Vessel fraction |", "|---|---|---|---|---|"]
        for d in sorted({r["dataset"] for r in lab}):
            xs = [r for r in lab if r["dataset"] == d]
            L.append(f"| {d} | {len(xs)} | {np.mean([float(r['width_px']) for r in xs]):.2f} | "
                     f"{1000 * np.mean([float(r['width_rel']) for r in xs]):.2f} | "
                     f"{np.mean([float(r['vessel_fraction']) for r in xs]):.4f} |")
    L += ["", "Width is the local diameter at centreline pixels (2 x distance to background - 1). The decision uses "
          "only the frozen models' outputs; label statistics are descriptive."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--external", default="data/external_FIVES")
    ap.add_argument("--out", default="results/label_style")
    ap.add_argument("--tta", action="store_true", help="8-fold TTA (about 8x slower)")
    ap.add_argument("--limit", type=int, default=None, help="images per model (quick test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        todo = jobs(args.runs, args.variants, args.seeds, args.external)
        if not todo:
            raise SystemExit(f"no checkpoints found under {args.runs}/")
        device = get_device(args.device)
        configure_backend(device, load_json(todo[0][1] / "config.json")["train"].get("gpu_memory_fraction"))  # shared GPU
        lstats = out / "label_stats.csv"
        if not lstats.exists():
            pool = Path(load_json(todo[0][1] / "config.json")["data"]["pool"])
            ext = Path(args.external)
            write_label_stats(lstats, [p if p.is_absolute() else ROOT / p for p in (pool, ext)], args.limit)
        path = out / "per_image.csv"
        done = set()
        if path.exists():
            with open(path) as f:
                done = {(r["source"], r["variant"], r["fold"], r["seed"], r["id"]) for r in csv.DictReader(f)}
        new = not path.exists()
        print(f"{len(todo)} models, TTA {'on' if args.tta else 'off'}, device {device}", flush=True)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            for n, (src, d) in enumerate(todo, 1):
                print(f"=== [{n}/{len(todo)}] {src} {d} ===", flush=True)
                run_model(src, d, args.external, args.tta, done, w, f, device, args.limit)
    summarize(out)


if __name__ == "__main__":
    main()
