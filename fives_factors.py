"""Exploratory: which measured image property goes with the hybrid - rgb loss on FIVES?

The darkening test showed that dimming an image does not reproduce the FIVES pattern, so brightness itself is not
the cause. This script measures other properties on the real images and asks which of them track the per-image gap
in Dice (hybrid - rgb, averaged over seeds) better than brightness does, and whether brightness still adds anything
once they are accounted for. Nothing is trained and no model is run. The labels are used only to locate vessels.

This looks at FIVES after its results were known, and tries many properties, so it can only suggest a cause to test
on data that does not involve FIVES. It is not a confirmatory result.

Measures (green channel, inside the FOV, at native resolution):
  brightness          mean green
  stretched_contrast  median over 384 px patches of (vessel contrast in the patch) / (patch max - patch min): how much
                      vessel contrast is left after the module's per-patch min-max. Dimming does not change it.
  vessel_contrast     (median green next to vessels - median green on vessels) / median green next to vessels
  bright_outlier      99.9th percentile / median green: optic disc and reflections relative to the rest of the eye
  sharpness           mean gradient on the vessel boundary / absolute vessel contrast, x FOV diameter / 1000
  noise               robust high-frequency spread away from vessels / median green
  shading             p95 / p5 of the heavily blurred green
  red_green, blue_green   mean red / mean green, mean blue / mean green
  vessel_fraction     vessel pixels / FOV pixels

Usage: python fives_factors.py [--workers 6]
"""
import argparse
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import rankdata, spearmanr

PATCH, OVERLAP = 384, 96
FACTORS = ["brightness", "stretched_contrast", "vessel_contrast", "bright_outlier", "sharpness", "noise", "shading",
           "red_green", "blue_green", "vessel_fraction"]


def read(folder, image_id, flag):
    path = next(iter(sorted(Path(folder).glob(image_id + ".*"))))
    return cv2.imread(str(path), flag)


def measure(job):
    root, image_id, fov_diameter = job
    bgr = read(f"{root}/images", image_id, cv2.IMREAD_COLOR)
    gt = read(f"{root}/masks", image_id, cv2.IMREAD_GRAYSCALE) > 0
    fov = read(f"{root}/fov", image_id, cv2.IMREAD_GRAYSCALE) > 0
    green = bgr[..., 1].astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner = cv2.erode(fov.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    vessel = gt & inner
    ring = (cv2.dilate(gt.astype(np.uint8), k) > 0) & ~gt & inner
    far = (cv2.dilate(gt.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) == 0) & inner
    v, r = np.median(green[vessel]), np.median(green[ring])
    abs_contrast = max(float(r - v), 1e-3)

    # vessel contrast left after the per-patch min-max, on the sliding-window grid used at inference
    ratios = []
    h, w = green.shape
    step = PATCH - OVERLAP
    for y in sorted({*range(0, max(1, h - PATCH), step), max(0, h - PATCH)}):
        for x in sorted({*range(0, max(1, w - PATCH), step), max(0, w - PATCH)}):
            g, ve, ri = green[y:y + PATCH, x:x + PATCH], vessel[y:y + PATCH, x:x + PATCH], ring[y:y + PATCH, x:x + PATCH]
            if ve.sum() < 200 or ri.sum() < 200:
                continue
            ratios.append((np.median(g[ri]) - np.median(g[ve])) / max(float(g.max() - g.min()), 1.0))

    gx, gy = cv2.Sobel(green, cv2.CV_32F, 1, 0) / 8, cv2.Sobel(green, cv2.CV_32F, 0, 1) / 8
    edge = (cv2.dilate(gt.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~(cv2.erode(gt.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & inner
    high = green - cv2.GaussianBlur(green, (0, 0), 1.5)
    hf = high[far]
    blurred = cv2.GaussianBlur(green, (0, 0), fov_diameter / 20)[inner]
    med = max(float(np.median(green[inner])), 1.0)
    mean_g = float(green[fov].mean())
    return {
        "id": image_id,
        "brightness": mean_g,
        "stretched_contrast": float(np.median(ratios)) if ratios else float("nan"),
        "vessel_contrast": abs_contrast / max(float(r), 1.0),
        "bright_outlier": float(np.percentile(green[inner], 99.9)) / med,
        "sharpness": float(np.hypot(gx, gy)[edge].mean()) / abs_contrast * fov_diameter / 1000,
        "noise": float(1.4826 * np.median(np.abs(hf - np.median(hf)))) / med,
        "shading": float(np.percentile(blurred, 95) / max(np.percentile(blurred, 5), 1.0)),
        "red_green": float(bgr[..., 2][fov].mean()) / max(mean_g, 1.0),
        "blue_green": float(bgr[..., 0][fov].mean()) / max(mean_g, 1.0),
        "vessel_fraction": float(gt[fov].mean()),
    }


def measure_all(root, ids, out, workers):
    meta = {r["id"]: r for r in csv.DictReader(open(f"{root}/metadata.csv"))}
    if Path(out).exists():
        rows = {r["id"]: r for r in csv.DictReader(open(out))}
        if set(ids) <= set(rows):
            return {i: {k: float(rows[i][k]) for k in FACTORS} for i in ids}
    with Pool(workers) as pool:
        got = pool.map(measure, [(root, i, float(meta[i]["fov_diameter"])) for i in ids], chunksize=4)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id"] + FACTORS)
        w.writeheader()
        w.writerows(got)
    return {r["id"]: {k: r[k] for k in FACTORS} for r in got}


def r2(X, y):
    X = np.column_stack([np.ones(len(y))] + X)
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - resid.var() / y.var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/fives_factors")
    args = ap.parse_args()

    dice, cond = defaultdict(dict), {}
    for r in csv.DictReader(open("results/stage3_vs_rgb/all_per_image.csv")):
        pass
    for variant in ("hybrid", "rgb", "clahe"):
        per = defaultdict(list)
        for seed in Path(f"runs/{variant}/all/all").glob("seed*"):
            for r in csv.DictReader(open(seed / "external_FIVES/per_image.csv")):
                if r["auc"] != "nan":
                    per[r["id"]].append(float(r["dice"]))
        for i, v in per.items():
            dice[variant][i] = np.mean(v)
    meta = {r["id"]: r for r in csv.DictReader(open("data/external_FIVES/metadata.csv"))}
    ids = sorted(dice["hybrid"])
    cond = np.array([meta[i]["condition"] for i in ids])
    gap = np.array([dice["hybrid"][i] - dice["rgb"][i] for i in ids])
    control = np.array([dice["clahe"][i] - dice["rgb"][i] for i in ids])

    fives = measure_all("data/external_FIVES", ids, f"{args.out}/fives_measures.csv", args.workers)
    split = __import__("json").load(open("splits/all.json"))[0]
    train = measure_all("data/pool", split["train"], f"{args.out}/train_measures.csv", args.workers)

    F = {k: np.array([fives[i][k] for i in ids]) for k in FACTORS}
    ok = np.all([np.isfinite(F[k]) for k in FACTORS], axis=0)
    print(f"{len(ids)} FIVES images, {ok.sum()} with every measure\n")
    gap_r, bright_r = rankdata(gap[ok]), rankdata(F["brightness"][ok])
    lines = ["| Property | with hybrid − rgb | within condition | after removing brightness | with clahe − rgb | R² alone | R² with brightness | FIVES outside training range |",
             "|---|---|---|---|---|---|---|---|"]
    for k in FACTORS:
        x, xr = F[k][ok], rankdata(F[k][ok])
        within = []
        for c in np.unique(cond):
            m = cond[ok] == c
            within.append(spearmanr(x[m], gap[ok][m])[0])
        resid = lambda a: a - np.polyval(np.polyfit(bright_r, a, 1), bright_r)
        partial = np.corrcoef(resid(xr), resid(gap_r))[0, 1] if k != "brightness" else float("nan")
        tr = np.array([train[i][k] for i in train if np.isfinite(train[i][k])])
        outside = 100 * np.mean((x < tr.min()) | (x > tr.max()))
        lines.append(f"| {k} | {spearmanr(x, gap[ok])[0]:+.2f} | {np.mean(within):+.2f} | {partial:+.2f} | "
                     f"{spearmanr(x, control[ok])[0]:+.2f} | {r2([xr], gap_r):.2f} | {r2([xr, bright_r], gap_r):.2f} | {outside:.0f}% |")
    all_r = [rankdata(F[k][ok]) for k in FACTORS]
    no_b = [rankdata(F[k][ok]) for k in FACTORS if k != "brightness"]
    cond_d = [(cond[ok] == c).astype(float) for c in np.unique(cond)[1:]]
    lines += ["", f"R² on ranks: all properties {r2(all_r, gap_r):.2f}; all except brightness {r2(no_b, gap_r):.2f}; "
              f"brightness alone {r2([bright_r], gap_r):.2f}; condition alone {r2(cond_d, gap_r):.2f}; "
              f"all properties plus condition {r2(all_r + cond_d, gap_r):.2f}", ""]

    lines += ["| Condition | n | hybrid − rgb | brightness | stretched_contrast | vessel_contrast | bright_outlier | sharpness | noise |", "|---|---|---|---|---|---|---|---|---|"]
    for c in np.unique(cond):
        m = cond == c
        lines.append(f"| {c} | {m.sum()} | {gap[m].mean():+.3f} | " + " | ".join(f"{np.nanmedian(F[k][m]):.3f}" for k in FACTORS[:2] + ["vessel_contrast", "bright_outlier", "sharpness", "noise"]) + " |")
    lines += ["", "Training images (100), median: " + ", ".join(f"{k} {np.nanmedian([train[i][k] for i in train]):.3f}" for k in FACTORS),
              "FIVES (median): " + ", ".join(f"{k} {np.nanmedian(F[k]):.3f}" for k in FACTORS)]
    text = "\n".join(lines)
    print(text)
    Path(args.out, "summary.md").write_text("# FIVES factors (exploratory)\n\n" + text + "\n")
    np.savez(f"{args.out}/gap.npz", ids=np.array(ids), gap=gap, cond=cond)


if __name__ == "__main__":
    main()
