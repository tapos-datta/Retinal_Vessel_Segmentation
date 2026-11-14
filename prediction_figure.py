"""Show what the models actually segment: per-pixel agreement with the reference, for rgb, hybrid and hybrid_weber.

  columns  RGB input | ground truth | rgb | hybrid | hybrid_weber
           prediction panels: true positive green, false positive red, false negative blue, over a faint grey image
  rows     LODO datasets (data/pool, seed-1 LODO model that held that dataset out) and FIVES (data/external_FIVES,
           seed-1 model trained on all 113 images). The showcase figure has CHASE_DB1, HRF and FIVES; the typical
           figure has one row per dataset: CHASE_DB1, DRIVE, HRF, STARE and FIVES.

Two figures:
  default    the showcase images of visualize_enhancement.py --final (CHASE_DB1 Image_14L, HRF 15_dr,
             FIVES_test_101_G), fixed in advance for the enhancement gallery; they show the enhancement's appearance
             and are not typical of the Dice means.          -> figures/predictions/
  --typical  per dataset, the image whose seed-1 hybrid_weber minus rgb Dice is closest to that dataset's median
             difference, chosen mechanically from the existing evaluate.py per_image.csv files.
                                                             -> figures/predictions/typical/

Predictions are made exactly as evaluate.py makes them: predict_sample (u2nete.infer) with 8-fold TTA, the run's
inference settings, threshold 0.5, resized back to the original resolution, scored inside the FOV with
u2nete.metrics.compute. Models are built by visualize_enhancement.load_variant from configs/variants/<variant>.json,
so hybrid_weber gets its Weber switch before the checkpoint is loaded (strict load).

Each panel's Dice is printed, written into its panel title and the caption, and checked against the run's
per_image.csv from evaluate.py.

Outputs: prediction_figure.png, prediction_caption.md, prediction_dice.json (plus typical_selection.json in --typical)

Usage:
    python prediction_figure.py --device cpu              # CPU, so the shared server GPU is left alone
    python prediction_figure.py --typical --device cpu
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

from u2nete.utils import ROOT, configure_backend, get_device, load_json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from u2nete.data import iter_samples, load_metadata
from u2nete.infer import predict_sample
from u2nete.io import read_binary, read_rgb
from u2nete.metrics import compute
from visualize_enhancement import GRID, INK, MUTED, SHOWCASE, load_variant

SEED = 1
ALL_SEEDS = [1, 2, 3]
VARIANTS = ["rgb", "hybrid", "hybrid_weber"]
DATASET_ROWS = [  # (row label, pool, split, fold, evaluate.py output folder); all five datasets (--typical)
    ("CHASE_DB1", "data/pool", "lodo", "lodo_CHASE_DB1", "test"),
    ("DRIVE", "data/pool", "lodo", "lodo_DRIVE", "test"),
    ("HRF", "data/pool", "lodo", "lodo_HRF", "test"),
    ("STARE", "data/pool", "lodo", "lodo_STARE", "test"),
    ("FIVES", "data/external_FIVES", "all", "all", "external_FIVES"),
]
SHOWCASE_IDS = {"CHASE_DB1": SHOWCASE["CHASE_DB1"], "HRF": SHOWCASE["HRF"], "FIVES": "FIVES_test_101_G"}
COLUMNS = ["RGB input", "Ground truth", *VARIANTS]
TP, FP, FN = (np.array(c) / 255.0 for c in ((26, 158, 92), (214, 39, 40), (42, 120, 214)))
THRESHOLD = 0.5
DISPLAY_PX = 1100  # longest side of each displayed panel


def faint_grey(rgb, fov):
    """Light grey version of the image (green channel inside the FOV squeezed into [0.72, 0.97], white outside),
    so the colours dominate."""
    g = rgb[..., 1].astype(np.float32) / 255.0
    lo, hi = np.percentile(g[fov > 0], [1, 99.5])
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    g = np.where(fov > 0, 0.72 + 0.25 * g, 1.0)
    return np.repeat(g[..., None], 3, axis=2)


def error_map(base, pred, gt, fov):
    """TP green, FP red, FN blue, inside the FOV only (pixels outside it are not scored)."""
    out = base.copy()
    fov, gt, pred = fov > 0, gt > 0, pred > 0
    out[pred & gt & fov] = TP
    out[pred & ~gt & fov] = FP
    out[~pred & gt & fov] = FN
    return out


def run_dir(runs, variant, split, fold, seed):
    return runs / variant / split / fold / f"seed{seed}"


def per_image_dice(path):
    """{image id: Dice} from an evaluate.py per_image.csv. Images with a blank reference label (auc = nan) cannot
    be scored and are left out, as in aggregate.py (FIVES_train_447_G and _448_G: FIVES n = 792, not 794)."""
    if not path.exists():
        raise SystemExit(f"missing {path} (run evaluate.py first)")
    with open(path) as fh:
        return {r["id"]: float(r["dice"]) for r in csv.DictReader(fh) if r["auc"] != "nan"}


def shrink(a):
    f = DISPLAY_PX / max(a.shape[:2])
    return cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else a


def select_typical(runs):
    """Per dataset: the image whose seed-1 (hybrid_weber - rgb) Dice is closest to the dataset's median difference
    (ties: smaller id). Also returns each variant's 3-seed mean Dice (mean over seeds of the per-seed dataset mean)."""
    out = {}
    for label, _, split, fold, out_name in DATASET_ROWS:
        d = {v: per_image_dice(run_dir(runs, v, split, fold, SEED) / out_name / "per_image.csv") for v in VARIANTS}
        ids = sorted(set.intersection(*(set(x) for x in d.values())))
        diff = {i: d["hybrid_weber"][i] - d["rgb"][i] for i in ids}
        median = float(np.median(list(diff.values())))
        pick = min(ids, key=lambda i: (abs(diff[i] - median), i))
        means = {}
        for v in VARIANTS:
            per_seed = [float(np.mean(list(per_image_dice(
                run_dir(runs, v, split, fold, s) / out_name / "per_image.csv").values()))) for s in ALL_SEEDS]
            means[v] = float(np.mean(per_seed))
        out[label] = {"id": pick, "n_images": len(ids), "median_diff": median, "image_diff": diff[pick],
                      "seed1_csv_dice": {v: d[v][pick] for v in VARIANTS}, "mean3_dice": means}
        print(f"{label:9} median hybrid_weber-rgb {median:+.4f} over {len(ids)} images -> {pick} "
              f"({diff[pick]:+.4f})   3-seed means " + ", ".join(f"{v} {means[v]:.4f}" for v in VARIANTS))
    return out


def predict_rows(rows, runs, device):
    """rows: [(label, pool, img_id, split, fold, out_name)]. Returns (panels, record, mismatches)."""
    panels, record, mismatches = [], [], []
    for label, pool, img_id, split, fold, out_name in rows:
        pool = Path(pool)
        pool = pool if pool.is_absolute() else ROOT / pool
        meta = load_metadata(pool)
        rgb = read_rgb(pool / "images" / f"{img_id}.png")
        gt = read_binary(pool / "masks" / f"{img_id}.png")
        fov = read_binary(pool / "fov" / f"{img_id}.png")
        base = faint_grey(rgb, fov)
        gt_view = base.copy()
        gt_view[gt > 0] = np.array([0.12, 0.12, 0.12])
        row = {"label": label, "id": img_id, "rgb": shrink(rgb), "gt": shrink(gt_view), "pred": {}, "dice": {}}

        for v in VARIANTS:
            rd = run_dir(runs, v, split, fold, SEED)
            cfg = load_json(rd / "config.json")  # what evaluate.py uses for data + inference settings
            if cfg["inference"]["threshold"] != THRESHOLD:
                raise SystemExit(f"{rd}: inference threshold {cfg['inference']['threshold']} != {THRESHOLD}")
            cache = ROOT / "results" / "prediction_figure" / "cache" / f"{v}_{split}_{fold}_seed{SEED}_{img_id}.npy"
            if cache.exists():
                prob = np.load(cache)
            else:
                model, _ = load_variant(v, rd / "best.pt", device)
                sample = next(iter_samples([img_id], pool, cfg["data"], meta))
                prob = predict_sample(model, sample, cfg, device, tta=True, to_original=True)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, prob.astype(np.float32))
                del model
            m = compute(prob, gt, fov, THRESHOLD, with_cldice=False)
            ref = per_image_dice(rd / out_name / "per_image.csv").get(img_id)
            note = "" if ref is None else f"  (evaluate.py {ref:.4f})"
            if ref is None or abs(ref - m["dice"]) > 5e-4:
                mismatches.append(f"{label} {v}: {m['dice']:.4f} vs evaluate.py {ref}")
            print(f"{label:9} {img_id:18} {v:13} Dice {m['dice']:.4f}  Prec {m['precision']:.4f}  "
                  f"Sens {m['sensitivity']:.4f}{note}")
            row["pred"][v] = shrink(error_map(base, prob >= THRESHOLD, gt, fov))
            row["dice"][v] = m["dice"]
            record.append({"row": label, "image": img_id, "variant": v, "checkpoint": str(rd / "best.pt"),
                           "dice": m["dice"], "precision": m["precision"], "sensitivity": m["sensitivity"],
                           "evaluate_py_dice": ref})
        panels.append(row)
    return panels, record, mismatches


def draw(panels, caption, title, path):
    fig, axes = plt.subplots(len(panels), len(COLUMNS), figsize=(3.4 * len(COLUMNS), 3.5 * len(panels)),
                             facecolor="white")
    for r, row in enumerate(panels):
        ims = [row["rgb"], row["gt"], *[row["pred"][v] for v in VARIANTS]]
        for c, im in enumerate(ims):
            ax = axes[r, c]
            ax.imshow(np.clip(im, 0, 1) if im.dtype != np.uint8 else im)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c >= 2:
                v = VARIANTS[c - 2]
                if r == 0:
                    ax.set_title(f"{v}\nDice {row['dice'][v]:.3f}", fontsize=12, color=INK, pad=8)
                else:
                    # inside the axes: panels differ in aspect ratio, so a title here can overlap the row above
                    ax.text(0.5, 0.995, f"Dice {row['dice'][v]:.3f}", transform=ax.transAxes, ha="center",
                            va="top", fontsize=11, color=INK,
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
            elif r == 0:
                ax.set_title(COLUMNS[c], fontsize=12, color=INK, pad=8)
            if c == 0:
                ax.set_ylabel(f"{row['label']}\n{row['id']}", fontsize=11, color=INK)
    fig.legend(handles=[Patch(color=TP, label="true positive"), Patch(color=FP, label="false positive"),
                        Patch(color=FN, label="false negative")],
               loc="upper right", ncol=3, frameon=False, fontsize=10, labelcolor=INK, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(title, fontsize=11, color=MUTED, x=0.01, ha="left", y=0.995)
    wrapped = "\n".join(textwrap.fill(p, int(fig.get_figwidth() * 72 / (9 * 0.62))) for p in caption.split("\n"))
    n_lines = wrapped.count("\n") + 1
    bottom = min(0.35, (0.16 * n_lines + 0.3) / fig.get_figheight())
    fig.tight_layout(rect=(0, bottom, 1, 0.975), h_pad=2.0)
    fig.text(0.01, bottom - 0.01, wrapped, ha="left", va="top", fontsize=9, color=INK, family="DejaVu Sans")
    fig.add_artist(plt.Line2D([0.01, 0.99], [bottom, bottom], color=GRID, linewidth=0.8))
    fig.savefig(path, dpi=130)
    plt.close(fig)


COLOUR_TEXT = ("Columns: RGB input, reference segmentation, then the predictions of rgb, hybrid and hybrid_weber. In "
               "the prediction panels green = true positive, red = false positive, blue = false negative, over a faint "
               "grey copy of the image; only pixels inside the FOV are coloured, because only those are scored.")
MODEL_TEXT = ("CHASE_DB1 and HRF are predicted by the seed-1 LODO model that held that dataset out; FIVES by the "
              "seed-1 model trained on all 113 pooled images.")
MODEL_TEXT_TYPICAL = ("CHASE_DB1, DRIVE, HRF and STARE are each predicted by the seed-1 LODO model that held that "
                      "dataset out; FIVES by the seed-1 model trained on all 113 pooled images.")
METHOD_TEXT = "Predictions as in evaluate.py: 8-fold TTA, threshold 0.5, at original resolution, Dice inside the FOV."


def panel_dice_text(row):
    return ", ".join(f"{v} {row['dice'][v]:.3f}" for v in VARIANTS)


def showcase_caption(panels):
    return "\n".join([
        COLOUR_TEXT,
        f"Rows: CHASE_DB1 Image_14L, HRF 15_dr and FIVES_test_101_G. {MODEL_TEXT} These images were fixed in advance "
        "as the showcase images of the enhancement gallery (visualize_enhancement.py --final), not chosen after seeing "
        "these predictions. They were chosen to show the enhancement's appearance and are not typical of the Dice "
        "means; see figures/predictions/typical/ for images chosen to be typical.",
        f"{METHOD_TEXT} Dice per panel: " + "; ".join(f"{r['label']} {r['id']}: {panel_dice_text(r)}"
                                                      for r in panels) + ".",
    ])


def typical_caption(panels, sel):
    lines = [
        COLOUR_TEXT,
        "Selection rule, applied mechanically to the existing evaluate.py per_image.csv files before these panels were "
        "drawn: for each dataset, the image whose hybrid_weber minus rgb Dice (seed 1, the models shown) is closest to "
        f"that dataset's median difference (ties: smaller id). Images with a blank reference label (AUC undefined) "
        f"are left out of the median and of the means, as in aggregate.py. {MODEL_TEXT_TYPICAL}",
        METHOD_TEXT,
    ]
    for r in panels:
        s = sel[r["label"]]
        lines.append(
            f"{r['label']} {r['id']}: hybrid_weber - rgb {s['image_diff']:+.3f} (dataset median {s['median_diff']:+.3f}, "
            f"n = {s['n_images']}). Panel Dice {panel_dice_text(r)}. Dataset 3-seed mean Dice "
            + ", ".join(f"{v} {s['mean3_dice'][v]:.3f}" for v in VARIANTS) + ".")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--typical", action="store_true",
                    help="per dataset, the image with the median hybrid_weber-rgb Dice difference "
                         "(writes to <out>/typical)")
    ap.add_argument("--runs", default="runs", help="root of the run folders")
    ap.add_argument("--out", default="figures/predictions")
    ap.add_argument("--device", default="auto",
                    help="auto (cuda, then mps, then cpu), or e.g. cpu / cuda:1; use cpu to leave a shared GPU alone")
    args = ap.parse_args()

    device = get_device(args.device)
    configure_backend(device)
    print(f"device: {device}")
    runs = Path(args.runs)
    out = Path(args.out) / ("typical" if args.typical else "")
    out.mkdir(parents=True, exist_ok=True)

    sel = select_typical(runs) if args.typical else None
    ids = {k: v["id"] for k, v in sel.items()} if sel else SHOWCASE_IDS
    rows = [(label, pool, ids[label], split, fold, out_name) for label, pool, split, fold, out_name in DATASET_ROWS
            if label in ids]  # the showcase figure keeps its three rows; --typical has all five datasets
    panels, record, mismatches = predict_rows(rows, runs, device)

    if args.typical:
        caption = typical_caption(panels, sel)
        title = "What each model segments on a typical image per dataset (seed 1, 8-fold TTA, threshold 0.5)"
        with open(out / "typical_selection.json", "w") as fh:
            json.dump({"rule": "argmin_i |(hybrid_weber - rgb Dice)_i - median|, seed 1, ties by id",
                       "seed": SEED, "mean_over_seeds": ALL_SEEDS, "datasets": sel}, fh, indent=2)
    else:
        caption = showcase_caption(panels)
        title = "What each model segments (seed 1, 8-fold TTA, threshold 0.5)"
    draw(panels, caption, title, out / "prediction_figure.png")
    (out / "prediction_caption.md").write_text(caption + "\n")
    with open(out / "prediction_dice.json", "w") as fh:
        json.dump({"seed": SEED, "tta": True, "threshold": THRESHOLD, "panels": record}, fh, indent=2)
    if mismatches:
        print("WARNING: Dice differs from evaluate.py per_image.csv:\n  " + "\n  ".join(mismatches))
    else:
        print("all panel Dice values match evaluate.py per_image.csv")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
