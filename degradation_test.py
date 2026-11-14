"""Can realistic image degradation reproduce the FIVES pattern without touching FIVES?

The darkening test dimmed images uniformly or with a smooth ramp, and hybrid hardly changed: its per-patch min-max
cancels a gain by construction. But dark FIVES images differ from the training images in more than mean brightness.
Measured medians (results/fives_factors, green channel inside the FOV), training images vs darkest FIVES quarter:

    brightness          59.8   vs 32.0
    bright_outlier       2.53  vs 5.48    p99.9 / median: optic disc and reflections stay bright while the rest is dark
    vessel_contrast      0.126 vs 0.107
    noise                0.0164 vs 0.0181 (relative to the median)
    sharpness            0.59  vs 0.86    FIVES is sharper, so blur and JPEG are not simulated (checked: sharpness
                                          is re-measured and should not fall)

Uniform dimming leaves bright_outlier, vessel_contrast and relative noise unchanged. This test changes them, alone
and together. Each step moves a property of an image by the factor that separates the darkest FIVES quarter from the
training images, measured on that image itself (bright_outlier x2.17, vessel_contrast x0.85, relative noise x1.10),
so images that already differ (CHASE_DB1 is noisier than FIVES) move by the same relative amount. Only image
statistics were used to set the levels; no FIVES labels or results. The test-image labels locate vessels when
vessel contrast is measured, as in fives_factors; they never reach the model. Each frozen LODO model (no training)
sees its own held-out images, and every variant sees exactly the same images.

Arms (each result is rounded to 8 bits; the random noise is fixed per image and level):
  disc_<g>       power curve that multiplies bright_outlier by 2.17 (optic disc and reflections stay bright while
                 the rest darkens), then a gain to mean green g (38, 32, 25; 32 is the darkest FIVES quarter's
                 median). Never brightens. A power curve also raises relative vessel contrast and noise.
  contrast_<p>   haze: blend in a very smooth copy (sigma = FOV diameter / 10, so vessel edges stay sharp) until vessel contrast is p % of
                 the original (85 matches FIVES, 70 is stronger). Mean brightness is kept.
  noise_<p>      add Gaussian noise until relative noise is p % of the original (110 matches FIVES; 150, 200)
  fives          disc_32, then contrast_85, then noise_110: matched to the darkest FIVES quarter (PRIMARY level)
  strong         disc_25, then contrast_70, then noise_150: beyond FIVES (secondary)
In the chains every target is relative to the original image. The power curve raises vessel contrast and the haze
lowers the bright outlier, so the two alternate three times to meet both; noise is added last, and only if the curve
has not already raised it enough. Blur and JPEG are not
simulated because FIVES is sharper than the training images (sharpness 0.86 vs 0.59).

Every degraded image is re-measured with the fives_factors definitions (calibration.csv); the summary shows each
property as a multiple of the original image, next to the factor FIVES shows.

Decision at the PRIMARY level `fives`, rule fixed in PROTOCOL.md before any result:
  (a) hybrid - rgb Dice <= -0.02 in each of the 3 seeds (same-seed means over the 113 images)
  (b) enh3 mean Dice lies strictly between hybrid and rgb (3-seed means)
  (c) hybrid misses vessels rather than drawing them thinner: hybrid - rgb cl_recall <= -0.02 and
      |hybrid - rgb width_ratio| <= |hybrid - rgb cl_recall|
  REPRODUCED      (a), (b) and (c): the level can stand in for FIVES when screening fixes
  PARTLY          (a) but not both (b) and (c)
  NOT REPRODUCED  (a) fails: these image properties do not explain hybrid's FIVES loss
Single-factor arms and `strong` are descriptive.

Usage (safe to stop and restart; finished predictions are kept):
    python degradation_test.py                        # 4 variants x 3 seeds x 4 folds, no TTA; about 3-4 h on an RTX 4090
    python degradation_test.py --calibrate-only       # measure the degraded images only (no models; minutes, CPU)
    python degradation_test.py --limit 2 --seeds 1    # quick check
    python degradation_test.py --summary-only
Writes results/degradation/per_image.csv, calibration.csv and summary.md.

Stage 6 ablation variants run in their own folder, on the same degraded images (the calibration is copied from the
reference folder), and are compared with the saved hybrid rows of the same seeds (rule R2 in PROTOCOL.md, stage 6):
    python degradation_test.py --variants hybrid_masknorm hybrid_rg hybrid_floor hybrid_weber --seeds 1 \
        --out results/degradation_stage6 --reference results/degradation
"""
import argparse
import csv
import zlib
from collections import defaultdict
from pathlib import Path

from u2nete.utils import ROOT, configure_backend, get_device, load_json

import cv2
import numpy as np
import torch
from scipy.stats import wilcoxon

from label_style_test import style_metrics
from u2nete.data import iter_samples, load_metadata
from u2nete.infer import predict_sample
from u2nete.metrics import compute
from u2nete.model import build_model

ORIGINAL = "original"
PRIMARY = "fives"
VARIANTS = ["hybrid", "rgb", "clahe", "enh3"]
LEVELS = [ORIGINAL, "disc_38", "disc_32", "disc_25", "contrast_85", "contrast_70",
          "noise_110", "noise_150", "noise_200", PRIMARY, "strong"]
CHAINS = {PRIMARY: ("disc_32", "contrast_85", "noise_110"), "strong": ("disc_25", "contrast_70", "noise_150")}
METRICS = ["dice", "cldice", "precision", "sensitivity", "cl_recall", "width_ratio"]
FIELDS = ["variant", "fold", "seed", "level", "id", "dataset", *METRICS]
FACTORS = ["brightness", "bright_outlier", "vessel_contrast", "stretched_contrast", "noise", "sharpness"]
# medians from results/fives_factors (see the docstring)
TARGETS = {"training": [59.80, 2.531, 0.1257, 0.0801, 0.0164, 0.590], "FIVES darkest quarter": [32.01, 5.484, 0.1071, 0.0624, 0.0181, 0.861]}
PATCH, OVERLAP = 384, 96
PARAMS = ["gamma", "haze", "gain"]


# ---------------------------------------------------------------- image measures (fives_factors definitions)

def _masks(gt, fov):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner = cv2.erode(fov.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    g = gt.astype(bool)
    vessel = g & inner
    ring = (cv2.dilate(g.astype(np.uint8), k) > 0) & ~g & inner
    far = (cv2.dilate(g.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) == 0) & inner
    return inner, vessel, ring, far


def _noise_abs(green, far):
    hf = (green - cv2.GaussianBlur(green, (0, 0), 1.5))[far]
    return float(1.4826 * np.median(np.abs(hf - np.median(hf))))


def measure(rgb, gt, fov, fov_diameter, masks=None):
    green = rgb[..., 1].astype(np.float32)
    inner, vessel, ring, far = masks or _masks(gt, fov)
    v, r = np.median(green[vessel]), np.median(green[ring])
    med = max(float(np.median(green[inner])), 1.0)
    ratios, (h, w), step = [], green.shape, PATCH - OVERLAP
    for y in sorted({*range(0, max(1, h - PATCH), step), max(0, h - PATCH)}):
        for x in sorted({*range(0, max(1, w - PATCH), step), max(0, w - PATCH)}):
            g, ve, ri = green[y:y + PATCH, x:x + PATCH], vessel[y:y + PATCH, x:x + PATCH], ring[y:y + PATCH, x:x + PATCH]
            if ve.sum() >= 200 and ri.sum() >= 200:
                ratios.append((np.median(g[ri]) - np.median(g[ve])) / max(float(g.max() - g.min()), 1.0))
    return {"brightness": float(green[fov > 0].mean()),
            "bright_outlier": float(np.percentile(green[inner], 99.9)) / med,
            "vessel_contrast": max(float(r - v), 1e-3) / max(float(r), 1.0),
            "stretched_contrast": float(np.median(ratios)) if ratios else float("nan"),
            "noise": _noise_abs(green, far) / med,
            "sharpness": _sharpness(green, gt, inner, max(float(r - v), 1e-3), fov_diameter)}


def _sharpness(green, gt, inner, abs_contrast, fov_diameter):
    g8 = gt.astype(np.uint8)
    edge = (cv2.dilate(g8, np.ones((3, 3), np.uint8)) > 0) & ~(cv2.erode(g8, np.ones((3, 3), np.uint8)) > 0) & inner
    gx, gy = cv2.Sobel(green, cv2.CV_32F, 1, 0) / 8, cv2.Sobel(green, cv2.CV_32F, 0, 1) / 8
    return float(np.hypot(gx, gy)[edge].mean()) / abs_contrast * fov_diameter / 1000


# ---------------------------------------------------------------- degradations
# Each step moves one property of an image by the factor that separates the darkest FIVES quarter from the training
# images (medians, results/fives_factors), measured on that image itself, so images that already differ (CHASE_DB1 is
# noisier than FIVES, DRIVE brighter) are moved by the same relative amount.
OUTLIER_X = 5.484 / 2.531    # bright_outlier x2.17
CONTRAST_X = 0.1071 / 0.1257  # vessel_contrast x0.85
NOISE_X = 0.0181 / 0.0164    # relative noise x1.10
GAMMA_MAX, HAZE_MAX = 2.5, 0.6  # beyond these the image stops looking like a photograph (vessels wash out); an image
                                # that needs more gets a partial match, counted in the summary
# The tone changes are solved on the green channel and applied to all three channels as one per-pixel factor, so the
# colour ratios stay as they were (red/green: training 2.50, darkest FIVES quarter 2.45).


def _u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _vc(green, masks):
    _, vessel, ring, _ = masks
    v, r = np.median(green[vessel]), np.median(green[ring])
    return max(float(r - v), 1e-3) / max(float(r), 1.0)


def _outlier(green, masks):
    inner = masks[0]
    return float(np.percentile(green[inner], 99.9)) / max(float(np.median(green[inner])), 1.0)


def _smooth(green, fov, fov_diameter):
    """Very smooth copy of the green channel inside the FOV (sigma = FOV diameter / 10, much wider than any vessel),
    without the black border leaking in. Blending towards it lowers vessel contrast but keeps vessel edges sharp."""
    inside = (fov > 0).astype(np.float32)
    sigma = max(fov_diameter / 10.0, 2.0)
    k = max(1, int(sigma // 8))  # blur a downsampled copy: same result for so wide a blur, far faster
    h, w = green.shape
    small = (max(1, w // k), max(1, h // k))
    num = cv2.GaussianBlur(cv2.resize(green * inside, small, interpolation=cv2.INTER_AREA), (0, 0), sigma / k)
    den = cv2.GaussianBlur(cv2.resize(inside, small, interpolation=cv2.INTER_AREA), (0, 0), sigma / k)
    return cv2.resize(num / np.maximum(den, 1e-3), (w, h), interpolation=cv2.INTER_LINEAR)


def _tone(g01, smooth01, gamma, haze):
    """Green after the power curve and the haze blend, both on 0-1 values (before the final gain)."""
    return (1 - haze) * g01 ** gamma + haze * smooth01 ** gamma


def _sub(x, k):
    return x[::k, ::k] if k > 1 else x


def solve(img, fov, steps, fov_diameter, masks):
    """Find gamma, haze fraction and gain on the green channel for the requested steps. Targets are relative to the
    image itself. Solved on a subsampled copy of large images (every k-th pixel) for speed."""
    k = max(1, int(fov_diameter // 1200) + 1)
    sm = tuple(_sub(m, k) for m in masks)
    g = img[..., 1].astype(np.float32)
    g01, s01 = _sub(g, k) / 255.0, _sub(_smooth(g, fov, fov_diameter), k) / 255.0
    r0, vc0 = _outlier(g01 * 255, sm), _vc(g01 * 255, sm)
    want_r = r0 * OUTLIER_X if "disc" in steps else None
    want_vc = vc0 * float(steps["contrast"]) / 100 if "contrast" in steps else None

    def haze_for(gamma):  # haze fraction that brings vessel contrast to the target at this gamma
        if want_vc is None or _vc(_tone(g01, s01, gamma, 0.0) * 255, sm) <= want_vc:
            return 0.0
        lo, hi = 0.0, HAZE_MAX
        for _ in range(14):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _vc(_tone(g01, s01, gamma, mid) * 255, sm) > want_vc else (lo, mid)
        return hi

    gamma, haze = 1.0, haze_for(1.0)
    if want_r is not None and r0 > 1.01:
        lo, hi = 1.0, GAMMA_MAX
        for _ in range(12):  # the bright outlier rises with gamma (the haze needed at each gamma is re-solved)
            mid = (lo + hi) / 2
            h = haze_for(mid)
            lo, hi = (mid, hi) if _outlier(_tone(g01, s01, mid, h) * 255, sm) < want_r else (lo, mid)
        gamma, haze = hi, haze_for(hi)
    mean0 = float(g[fov > 0].mean())
    gain = 1.0
    if "disc" in steps:  # gain to the target mean green; never brighter than the original image
        full = _tone(g / 255.0, _smooth(g, fov, fov_diameter) / 255.0, gamma, haze) * 255
        gain = min(float(steps["disc"]), mean0) / max(float(full[fov > 0].mean()), 1e-6)
    return {"gamma": gamma, "haze": haze, "gain": gain}


def apply_tone(img, fov, params, fov_diameter):
    if params["gamma"] == 1.0 and params["haze"] == 0.0 and params["gain"] == 1.0:
        return img
    f = img.astype(np.float32)
    g = f[..., 1]
    new_g = _tone(g / 255.0, _smooth(g, fov, fov_diameter) / 255.0, params["gamma"], params["haze"]) * 255 * params["gain"]
    factor = (new_g + 1.0) / (g + 1.0)
    out = _u8(f * factor[..., None])
    out[fov == 0] = img[fov == 0]
    return out


def add_noise(img, fov, target_rel, seed, masks):
    """Add Gaussian noise (same sigma in every channel) until measured relative noise reaches target_rel."""
    inner, _, _, far = masks
    green = img[..., 1].astype(np.float32)
    need = target_rel * max(float(np.median(green[inner])), 1.0)
    have = _noise_abs(green, far)
    if have >= need:
        return img
    unit = np.random.default_rng(seed).standard_normal(img.shape).astype(np.float32)
    s = np.sqrt(need ** 2 - have ** 2) / _HP_GAIN
    for _ in range(6):  # rounding to 8 bits changes the measured noise a little; correct for it
        out = _u8(img.astype(np.float32) + s * unit)
        got = _noise_abs(out[..., 1].astype(np.float32), far)
        if abs(got - need) < 0.02 * need:
            break
        s *= np.sqrt(max(need ** 2 - have ** 2, 1e-6) / max(got ** 2 - have ** 2, 1e-6))
    out[fov == 0] = img[fov == 0]
    return out


def _hp_gain():
    """Fraction of white-noise spread that survives the noise measure's high-pass."""
    z = np.random.default_rng(0).standard_normal((512, 512)).astype(np.float32)
    hf = (z - cv2.GaussianBlur(z, (0, 0), 1.5))[32:-32, 32:-32]
    return float(1.4826 * np.median(np.abs(hf - np.median(hf))))


_HP_GAIN = _hp_gain()


def degrade(img, gt, fov, level, image_id, fov_diameter, masks, params=None):
    """Returns (degraded image, tone parameters). Pass cached `params` to skip solving."""
    if level == ORIGINAL:
        return img, {}
    steps = dict(st.split("_") for st in CHAINS.get(level, (level,)))
    if params is None:
        params = solve(img, fov, steps, fov_diameter, masks) if ("disc" in steps or "contrast" in steps) else {}
    out = apply_tone(img, fov, params, fov_diameter) if params else img
    if "noise" in steps:
        g0 = img[..., 1].astype(np.float32)
        n0 = _noise_abs(g0, masks[3]) / max(float(np.median(g0[masks[0]])), 1.0)
        out = add_noise(out, fov, n0 * float(steps["noise"]) / 100, zlib.crc32(f"{image_id}/{level}".encode()), masks)
    return out, params


# ---------------------------------------------------------------- running

def lodo_dirs(runs, variants, seeds):
    return [d for v in variants for s in seeds
            for d in sorted(Path(runs).glob(f"{v}/lodo/*/seed{s}")) if (d / "best.pt").exists()]


def pool_of(cfg):
    p = Path(cfg["data"]["pool"])
    return p if p.is_absolute() else ROOT / p


def calibrate(dirs, levels, out, limit):
    """Measure every degraded test image once (independent of the model)."""
    path = out / "calibration.csv"
    done = set()
    if path.exists():
        with open(path) as f:
            done = {(r["level"], r["id"]) for r in csv.DictReader(f)}
    seen = {}
    for d in dirs:  # each image is a LODO test image of exactly one fold; any seed and variant gives the same list
        cfg = load_json(d / "config.json")
        for i in load_json(d / "fold.json")["test"][:limit]:
            seen.setdefault(i, cfg)
    todo = [i for i in sorted(seen) if any((lv, i) not in done for lv in levels)]
    if not todo:
        return
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level", "id", "dataset", *FACTORS, *PARAMS])
        if new:
            w.writeheader()
        cfg = seen[todo[0]]
        meta = load_metadata(pool_of(cfg))
        for s in iter_samples(todo, pool_of(cfg), cfg["data"], meta):
            masks = _masks(s.gt, s.fov)
            for lv in levels:
                if (lv, s.id) in done:
                    continue
                d = float(meta[s.id]["fov_diameter"]) * s.scale
                img, params = degrade(s.image, s.gt, s.fov, lv, s.id, d, masks)
                m = measure(img, s.gt, s.fov, d, masks)
                w.writerow({"level": lv, "id": s.id, "dataset": s.dataset, **{k: f"{m[k]:.5f}" for k in FACTORS},
                            **{k: repr(params.get(k, "")) if k in params else "" for k in PARAMS}})
            f.flush()
            print(f"  calibrated {s.id}", flush=True)


def load_params(out):
    """Tone parameters solved during calibration, so every model sees exactly the same degraded images."""
    path = out / "calibration.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        return {(r["level"], r["id"]): {k: float(r[k]) for k in PARAMS if r.get(k)} for r in csv.DictReader(f)}


def run_model(run_dir, levels, done, writer, handle, device, limit, cached):
    cfg = load_json(run_dir / "config.json")
    pool = pool_of(cfg)
    meta = load_metadata(pool)
    ids = load_json(run_dir / "fold.json")["test"][:limit]
    variant, _, fold, seed = run_dir.parts[-4:]
    todo = [i for i in ids if any((variant, fold, seed, lv, i) not in done for lv in levels)]
    if not todo:
        return
    ck = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    thr = cfg["inference"]["threshold"]
    for s in iter_samples(todo, pool, cfg["data"], meta):
        original, masks = s.image, _masks(s.gt, s.fov)
        for lv in levels:
            if (variant, fold, seed, lv, s.id) in done:
                continue
            s.image, _ = degrade(original, s.gt, s.fov, lv, s.id, float(meta[s.id]["fov_diameter"]) * s.scale, masks,
                                 cached.get((lv, s.id)))
            prob = predict_sample(model, s, cfg, device, tta=False, to_original=True)
            m = compute(prob, s.gt, s.fov, thr)
            m.update(style_metrics(prob >= thr, s.gt, s.fov))
            writer.writerow({"variant": variant, "fold": fold, "seed": seed, "level": lv, "id": s.id,
                             "dataset": s.dataset, **{k: f"{m[k]:.5f}" for k in METRICS}})
            handle.flush()
        s.image = original
        print(f"  {variant:7} {fold:16} {seed} {s.id:14} done", flush=True)
    del model


# ---------------------------------------------------------------- summary

def merge_reference(rows, reference):
    """Add saved rows of variants not tested here, for the same seeds, images and levels."""
    with open(Path(reference) / "per_image.csv") as f:
        ref = list(csv.DictReader(f))
    tested, seeds, ids, lvls = ({r[k] for r in rows} for k in ("variant", "seed", "id", "level"))
    return rows + [r for r in ref if r["variant"] not in tested and r["seed"] in seeds
                   and r["id"] in ids and r["level"] in lvls]


R2_LEVELS, R2_MARGIN = ("disc_32", PRIMARY), 0.010


def summarize(out, levels, reference=None):
    L = ["# Degradation test", "",
         "Frozen LODO models on their own held-out images, no TTA, threshold 0.5. Mean over the 113 images, "
         "averaged over seeds. Rule in PROTOCOL.md (\"Degradation test\").", ""]
    cpath = out / "calibration.csv"
    if cpath.exists():
        with open(cpath) as f:
            cal = list(csv.DictReader(f))
        L += ["## Calibration: measured image properties", "",
              "Median over images. Left: absolute values; right: value / the same image's original value.", "",
              "| Level | " + " | ".join(FACTORS) + " | " + " | ".join(f"{k} ×" for k in FACTORS[:3] + FACTORS[4:]) + " |",
              "|---|" + "---|" * (2 * len(FACTORS) - 1)]
        ratio_keys = FACTORS[:3] + FACTORS[4:]
        tr, fv = TARGETS["training"], TARGETS["FIVES darkest quarter"]
        L.append("| *training → FIVES darkest quarter* | " + " | ".join(f"*{a:.4g} → {b:.4g}*" for a, b in zip(tr, fv)) + " | "
                 + " | ".join(f"*{fv[FACTORS.index(k)] / tr[FACTORS.index(k)]:.2f}*" for k in ratio_keys) + " |")
        orig = {(r["id"], k): float(r[k]) for r in cal if r["level"] == ORIGINAL for k in FACTORS}
        for lv in levels:
            xs = [r for r in cal if r["level"] == lv]
            if not xs:
                continue
            absv = [f"{np.nanmedian([float(r[k]) for r in xs]):.4g}" for k in FACTORS]
            rel = [f"{np.nanmedian([float(r[k]) / orig[(r['id'], k)] for r in xs if orig.get((r['id'], k))]):.2f}"
                   if orig else "–" for k in ratio_keys]
            L.append(f"| {lv} | " + " | ".join(absv + rel) + " |")
        capped = {lv: sum(1 for r in cal if r["level"] == lv and r.get("gamma") and
                          (float(r["gamma"]) >= GAMMA_MAX - 1e-3 or float(r["haze"]) >= HAZE_MAX - 1e-3))
                  for lv in levels}
        if any(capped.values()):
            L += ["", "Images whose tone change reached a limit (power curve " + f"{GAMMA_MAX}, haze {HAZE_MAX}" +
                  ") and so match the targets only partly: " + ", ".join(f"{lv} {n}" for lv, n in capped.items() if n)]
        L.append("")

    ppath = out / "per_image.csv"
    if not ppath.exists():
        (out / "summary.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    with open(ppath) as f:
        rows = list(csv.DictReader(f))
    if reference:
        rows = merge_reference(rows, reference)
        L += [f"Rows of variants not run here come from `{reference}`, limited to the same seeds, images and levels.", ""]
    added = sorted({r["variant"] for r in rows} - set(VARIANTS))
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)] + added
    seeds = sorted({r["seed"] for r in rows})
    keys = [lv for lv in levels if any(r["level"] == lv for r in rows)]
    img = defaultdict(list)       # (variant, level, id, metric) -> values over seeds
    seed_vals = defaultdict(dict)  # (variant, level, seed, metric) -> {id: value}
    for r in rows:
        for k in METRICS:
            x = float(r[k])
            img[(r["variant"], r["level"], r["id"], k)].append(x)
            seed_vals[(r["variant"], r["level"], r["seed"], k)][r["id"]] = x
    ids = sorted({r["id"] for r in rows})

    def mean(v, lv, k, subset=None):
        xs = [np.nanmean(img[(v, lv, i, k)]) for i in (subset or ids) if (v, lv, i, k) in img]
        return float(np.nanmean(xs)) if xs else float("nan")

    def paired(a, b, lv, k):
        common = [i for i in ids if (a, lv, i, k) in img and (b, lv, i, k) in img]
        return np.array([np.nanmean(img[(a, lv, i, k)]) - np.nanmean(img[(b, lv, i, k)]) for i in common])

    def seed_diff(a, b, lv, k, s):
        x, y = seed_vals[(a, lv, s, k)], seed_vals[(b, lv, s, k)]
        common = sorted(set(x) & set(y))
        return float(np.nanmean([x[i] - y[i] for i in common])) if common else float("nan")

    L += ["## Mean Dice by level", "", "| Level | " + " | ".join(variants) +
          " | hybrid − rgb | better / worse | p | enh3 − rgb |", "|---|" + "---|" * (len(variants) + 4)]
    for lv in keys:
        cells = [f"{mean(v, lv, 'dice'):.4f}" for v in variants]
        extra = []
        if {"hybrid", "rgb"} <= set(variants):
            x = paired("hybrid", "rgb", lv, "dice")
            p = wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else float("nan")
            extra += [f"{x.mean():+.4f}", f"{(x > 0).sum()} / {(x < 0).sum()}", f"{p:.2g}"]
        else:
            extra += ["–"] * 3
        extra.append(f"{paired('enh3', 'rgb', lv, 'dice').mean():+.4f}" if {"enh3", "rgb"} <= set(variants) else "–")
        L.append(f"| {lv} | " + " | ".join(cells + extra) + " |")

    if {"hybrid", "rgb"} <= set(variants):
        L += ["", "## hybrid − rgb: finding vessels vs drawing their width", "",
              "| Level | Dice | cl_recall | width_ratio | sensitivity | precision | Dice by seed |", "|---|---|---|---|---|---|---|"]
        for lv in keys:
            d = {k: paired("hybrid", "rgb", lv, k) for k in ("dice", "cl_recall", "width_ratio", "sensitivity", "precision")}
            by_seed = ", ".join(f"{seed_diff('hybrid', 'rgb', lv, 'dice', s):+.4f}" for s in seeds)
            L.append(f"| {lv} | " + " | ".join(f"{np.nanmean(d[k]):+.4f}" for k in d) + f" | {by_seed} |")

        datasets = sorted({r["dataset"] for r in rows})
        ds_ids = {d: {r["id"] for r in rows if r["dataset"] == d} for d in datasets}
        L += ["", "## hybrid − rgb Dice by held-out dataset", "", "| Level | " + " | ".join(datasets) + " |",
              "|---|" + "---|" * len(datasets)]
        for lv in keys:
            L.append(f"| {lv} | " + " | ".join(
                f"{mean('hybrid', lv, 'dice', ds_ids[d]) - mean('rgb', lv, 'dice', ds_ids[d]):+.4f}" for d in datasets) + " |")

    if PRIMARY in keys and {"hybrid", "rgb"} <= set(variants) and cpath.exists():
        capped_ids = {r["id"] for r in cal if r["level"] == PRIMARY and r.get("gamma") and
                      (float(r["gamma"]) >= GAMMA_MAX - 1e-3 or float(r["haze"]) >= HAZE_MAX - 1e-3)}
        L += ["", f"## hybrid − rgb at `{PRIMARY}`, by calibration match (descriptive)", "",
              "| Images | n | Dice | cl_recall | width_ratio |", "|---|---|---|---|---|"]
        for name, sub in (("reached a limit (partial match)", capped_ids), ("full match", set(ids) - capped_ids)):
            sub = sorted(set(ids) & sub)
            if sub:
                L.append(f"| {name} | {len(sub)} | " + " | ".join(
                    f"{mean('hybrid', PRIMARY, k, sub) - mean('rgb', PRIMARY, k, sub):+.4f}"
                    for k in ("dice", "cl_recall", "width_ratio")) + " |")

    if added and "hybrid" in variants and ORIGINAL in keys:
        L += ["", "## Stage 6: robustness of each variant against hybrid (rule R2)", "",
              "Dice lost from the variant's own original images, against hybrid's loss on the same images and seeds. "
              f"R2 holds if the loss is at most hybrid's + {R2_MARGIN} at {' and '.join(R2_LEVELS)}.", "",
              "| Variant | Original | " + " | ".join(f"{lv} loss (hybrid)" for lv in keys if lv != ORIGINAL) + " | R2 |",
              "|---|---|" + "---|" * (len(keys) - 1) + "---|"]
        for v in ["hybrid"] + added:
            o_v, o_h = mean(v, ORIGINAL, "dice"), mean("hybrid", ORIGINAL, "dice")
            cells, ok = [], True
            for lv in keys:
                if lv == ORIGINAL:
                    continue
                dv, dh = mean(v, lv, "dice") - o_v, mean("hybrid", lv, "dice") - o_h
                cells.append(f"{dv:+.4f} ({dh:+.4f})")
                if lv in R2_LEVELS and not dv >= dh - R2_MARGIN:
                    ok = False
            r2 = "–" if v == "hybrid" else ("yes" if ok and all(lv in keys for lv in R2_LEVELS) else "no")
            L.append(f"| {v} | {o_v:.4f} | " + " | ".join(cells) + f" | {r2} |")

    if PRIMARY in keys and {"hybrid", "rgb", "enh3"} <= set(variants) and not added:
        per_seed = [seed_diff("hybrid", "rgb", PRIMARY, "dice", s) for s in seeds]
        a = len(seeds) == 3 and all(v <= -0.02 for v in per_seed)
        h, r, e = (mean(v, PRIMARY, "dice") for v in ("hybrid", "rgb", "enh3"))
        b = min(h, r) < e < max(h, r)
        dcl = float(np.nanmean(paired("hybrid", "rgb", PRIMARY, "cl_recall")))
        dw = float(np.nanmean(paired("hybrid", "rgb", PRIMARY, "width_ratio")))
        c = dcl <= -0.02 and abs(dw) <= abs(dcl)
        verdict = "REPRODUCED" if a and b and c else "PARTLY" if a else "NOT REPRODUCED"
        complete = len(seeds) == 3 and len(ids) == 113 and all(
            len(seed_vals[(v, PRIMARY, s, "dice")]) == 113 for v in ("hybrid", "rgb", "enh3") for s in seeds)
        L += ["", f"## Decision (level `{PRIMARY}`)", "",
              f"- (a) hybrid − rgb Dice by seed: {', '.join(f'{v:+.4f}' for v in per_seed)}; all ≤ −0.02: **{'yes' if a else 'no'}**",
              f"- (b) enh3 {e:.4f} between hybrid {h:.4f} and rgb {r:.4f}: **{'yes' if b else 'no'}**",
              f"- (c) hybrid − rgb cl_recall {dcl:+.4f}, width_ratio {dw:+.4f}: **{'yes' if c else 'no'}**", "",
              f"**{verdict}**" + ("" if complete else " (INCOMPLETE: not all runs have finished)")]
    L += ["", "Paired tests use per-image values averaged over seeds; they treat images, not seeds, as the sample."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--levels", nargs="*", default=LEVELS)
    ap.add_argument("--out", default="results/degradation")
    ap.add_argument("--limit", type=int, default=None, help="images per fold (quick test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--save-examples", type=int, default=0, help="write this many example images per level as PNG")
    ap.add_argument("--reference", help="folder of an earlier run: its calibration is reused and its rows are compared")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.reference and not (out / "calibration.csv").exists():
        import shutil
        shutil.copy(Path(args.reference) / "calibration.csv", out / "calibration.csv")
    levels = [lv for lv in LEVELS if lv in args.levels]
    if not args.summary_only:
        dirs = lodo_dirs(args.runs, args.variants, args.seeds)
        if not dirs:
            raise SystemExit(f"no LODO checkpoints found under {args.runs}/")
        calibrate(dirs, levels, out, args.limit)
        if args.save_examples:
            save_examples(dirs, levels, out / "examples", args.save_examples, load_params(out))
        if not args.calibrate_only:
            path = out / "per_image.csv"
            done = set()
            if path.exists():
                with open(path) as f:
                    done = {(r["variant"], r["fold"], r["seed"], r["level"], r["id"]) for r in csv.DictReader(f)}
            device = get_device(args.device)
            configure_backend(device, load_json(dirs[0] / "config.json")["train"].get("gpu_memory_fraction"))
            cached = load_params(out)
            print(f"{len(dirs)} models, {len(levels)} levels, no TTA, device {device}", flush=True)
            new = not path.exists()
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                for n, d in enumerate(dirs, 1):
                    print(f"=== [{n}/{len(dirs)}] {d} ===", flush=True)
                    run_model(d, levels, done, w, f, device, args.limit, cached)
    summarize(out, levels, args.reference)


def save_examples(dirs, levels, folder, n, cached):
    """One image per held-out dataset (up to n), every level side by side, for a visual check."""
    folder.mkdir(parents=True, exist_ok=True)
    picked = {}
    for d in dirs:
        cfg = load_json(d / "config.json")
        ids = load_json(d / "fold.json")["test"]
        if ids and len(picked) < n:
            picked.setdefault(d.parts[-2], (ids[0], cfg))
    for _, (i, cfg) in picked.items():
        meta = load_metadata(pool_of(cfg))
        s = next(iter_samples([i], pool_of(cfg), cfg["data"], meta))
        masks = _masks(s.gt, s.fov)
        tiles = []
        for lv in levels:
            img, _ = degrade(s.image, s.gt, s.fov, lv, s.id, float(meta[s.id]["fov_diameter"]) * s.scale, masks,
                             cached.get((lv, s.id)))
            t = cv2.resize(img, (360, round(360 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.putText(t, lv, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(t)
        rows = [np.hstack(tiles[k:k + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[k:k + 4])))
                for k in range(0, len(tiles), 4)]
        cv2.imwrite(str(folder / f"{s.dataset}_{i}.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
