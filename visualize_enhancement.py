"""Show what the learnable enhancement front-end does, and whether it makes the four datasets look more alike.

For every pooled image the enhancement module of a trained checkpoint is run on its own (sliding 384 px
patches, exactly as the network sees them) and compared with two fixed inputs: the raw green channel and
CLAHE on the green channel.

Outputs (in --out):
  enhancement_gallery.png   one image per dataset: RGB | green | CLAHE | U2-Net-E enhanced (shared display scale)
  enhancement_zoom.png      the same, zoomed on a vessel-dense region (same physical size in every dataset)
  enhancement_histograms.png  background-intensity distribution per dataset, per representation
  enhancement_contrast.png  vessel contrast-to-noise ratio per dataset, per representation
  enhancement_metrics.csv   per-image numbers; enhancement_summary.json  per-dataset numbers + dataset gap

Final-model mode (--final) compares the original module with the final Part 1 model, one row per dataset:
  columns  RGB | green | CLAHE (green) | hybrid | hybrid_weber   (one shared grey scale per column)
  rows     CHASE_DB1, DRIVE, HRF, STARE (seed-1 LODO model that held that dataset out) and FIVES_test_101_G
           (seed-1 model trained on all 113 images); both enhanced columns of a row come from the same fold and seed
  outputs  enhancement_final_gallery.png, enhancement_final_zoom.png, enhancement_final_caption.md,
           enhancement_final_checkpoints.json (in --out, default figures/enhancement_final)
Models are built from configs/variants/<variant>.json, so hybrid_weber gets set_ablation(norm="weber") before its
checkpoint is loaded (its state dict has the Weber illumination and gain parameters).

Usage:
    python visualize_enhancement.py --checkpoint ../runs/run1_hybrid_norm_true/checkpoints_HYBRID_4SET/best_model_epoch_3834_val_0.1330.pth
    python visualize_enhancement.py --checkpoint runs/hybrid_scale_strong/lodo/lodo_HRF/seed1/best.pt --out figures/v2_lodo_HRF
    python visualize_enhancement.py --final --device cpu        # CPU, so the shared server GPU is left alone
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

from u2nete.utils import get_device, load_config, save_json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from u2nete.data import fov_mask_image, fov_normalize, load_metadata, preprocess, resize_sample, working_scale
from u2nete.io import read_binary, read_rgb
from u2nete.model import build_model
from u2nete.models.U2NetArc import U2NET_lite
from u2nete.models.U2NetE import U2NetE

DATASETS = ["CHASE_DB1", "DRIVE", "HRF", "STARE"]
COLORS = {"CHASE_DB1": "#2a78d6", "DRIVE": "#eb6834", "HRF": "#1baf7a", "STARE": "#eda100"}  # validated, fixed order
REPS = ["green", "clahe", "enhanced"]
REP_LABEL = {"green": "Green channel", "clahe": "CLAHE (green)", "enhanced": "U²-Net-E enhanced"}
SHOWCASE = {"CHASE_DB1": "Image_14L", "DRIVE": "39_training", "HRF": "15_dr", "STARE": "im0319"}  # run1 test images
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

# --final: original module vs the final Part 1 model, seed 1 throughout
FINAL_SEED = 1
FINAL_VARIANTS = ["hybrid", "hybrid_weber"]
FINAL_ROWS = [  # (row label, pool, image id, split, fold)
    *[(ds, "data/pool", SHOWCASE[ds], "lodo", f"lodo_{ds}") for ds in DATASETS],
    ("FIVES", "data/external_FIVES", "FIVES_test_101_G", "all", "all"),
]
FINAL_COLS = [("rgb", "RGB input"), ("green", "Green channel"), ("clahe", "CLAHE (green)"),
              ("hybrid", "hybrid"), ("hybrid_weber", "hybrid_weber")]


def load_model(path, device):
    """Accepts run1/run2 checkpoints and v2 best.pt files. Returns (model, fov_norm, data_cfg)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "config" in ck:  # v2 checkpoint carries its config
        cfg = ck["config"]
        if cfg["model"]["enhancement"] not in ("hybrid", "v2", "v3"):
            raise SystemExit("this checkpoint has no learnable enhancement module to visualize")
        model = build_model(cfg["model"])
        model.load_state_dict(ck["model"])
        return model.to(device).eval(), bool(cfg["data"].get("fov_norm", False)), cfg["data"]
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model = U2NetE(U2NET_lite(), enhancement_mode="hybrid", normalize="mean" in sd)
    model.load_state_dict(sd)
    return model.to(device).eval(), False, {}


def load_variant(variant, path, device):
    """Build a model from configs/variants/<variant>.json, then load a v2 checkpoint into it.

    build_model applies the config's ablation switch (hybrid_weber: set_ablation(norm="weber")) while constructing the
    module, i.e. before load_state_dict, so the Weber parameters exist and the strict load matches.
    Returns (model, data_cfg)."""
    cfg = load_config(Path("configs/variants") / f"{variant}.json")
    model = build_model(cfg["model"])
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "config" in ck):
        raise SystemExit(f"{path}: not a v2 checkpoint (no stored config)")
    stored = ck["config"]["model"]
    for key in ("enhancement", "ablation"):
        if stored.get(key) != cfg["model"].get(key):
            raise SystemExit(f"{path}: checkpoint model.{key}={stored.get(key)!r} does not match "
                             f"configs/variants/{variant}.json ({cfg['model'].get(key)!r})")
    if variant == "hybrid_weber" and getattr(model.enhance, "norm", None) != "weber":
        raise SystemExit("hybrid_weber was built without the Weber switch; check configs/variants/hybrid_weber.json")
    model.load_state_dict(ck["model"])  # strict
    return model.to(device).eval(), cfg["data"]


@torch.no_grad()
def enhance_full(model, rgb, device, patch=384, overlap=64, batch=16, fov=None):
    """Run only the enhancement module over the image, exactly as inference does.

    EnhancementV2/V3 work on the whole image at once; the original module runs per 384 px patch (and
    min-max stretches each patch, which is what produces its tiles).
    """
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
    accepts_fov = getattr(model.enhance, "accepts_fov", False)
    if hasattr(model.enhance, "enhance_image") and (
        getattr(model.enhance, "supports_any_size", False) or x.shape[1] * x.shape[2] <= 6_000_000
    ):
        if accepts_fov and fov is not None:
            fov_t = torch.from_numpy(fov.astype(np.float32))[None, None].to(device)
            return model.enhance.enhance_image(x[None].to(device), patch, fov=fov_t)[0, 0].cpu().numpy()
        return model.enhance.enhance_image(x[None].to(device), patch)[0, 0].cpu().numpy()
    _, H, W = x.shape
    ph, pw = max(0, patch - H), max(0, patch - W)
    if ph or pw:
        x = torch.nn.functional.pad(x[None], (0, pw, 0, ph), mode="reflect")[0]
    _, Hp, Wp = x.shape
    stride = patch - overlap
    ys = list(range(0, Hp - patch + 1, stride)) + [Hp - patch]
    xs = list(range(0, Wp - patch + 1, stride)) + [Wp - patch]
    coords = sorted({(y, x0) for y in ys for x0 in xs})
    acc, cnt = torch.zeros(Hp, Wp), torch.zeros(Hp, Wp)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        out = model.enhance(torch.stack([x[:, y:y + patch, x0:x0 + patch] for y, x0 in chunk]).to(device))[:, 0].cpu()
        for (y, x0), o in zip(chunk, out):
            acc[y:y + patch, x0:x0 + patch] += o
            cnt[y:y + patch, x0:x0 + patch] += 1
    return (acc / cnt)[:H, :W].numpy()


def representations(model, rgb, fov, device, fov_norm=False, mask_fov=False):
    model_rgb = fov_normalize(rgb, fov) if fov_norm else (fov_mask_image(rgb, fov) if mask_fov else rgb)
    return {
        "green": rgb[..., 1].astype(np.float32) / 255.0,
        "clahe": preprocess(rgb, "clahe")[..., 0].astype(np.float32) / 255.0,
        "enhanced": enhance_full(model, model_rgb, device, fov=fov),
    }


def contrast_stats(img, gt, fov):
    v = img[(gt > 0) & (fov > 0)]
    b = img[(gt == 0) & (fov > 0)]
    sd = float(b.std()) + 1e-8
    return {"vessel_median": float(np.median(v)), "background_median": float(np.median(b)),
            "background_sd": sd, "cnr": float((b.mean() - v.mean()) / sd)}


def vessel_dense_centre(gt, fov, win):
    k = np.ones((win, win), np.float32) / (win * win)
    dens = cv2.filter2D(gt.astype(np.float32), -1, k) * cv2.erode(fov, np.ones((win, win), np.uint8))
    y, x = np.unravel_index(np.argmax(dens), dens.shape)
    return int(y), int(x)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def gallery(rows, ranges, path, zoom=False, columns=None, title=None, caption=None):
    """columns: [(key, label), ...]; default is the single-checkpoint layout RGB | green | CLAHE | enhanced.
    title / caption replace the default suptitle and add a wrapped caption under the figure (final mode)."""
    columns = columns or [("rgb", "RGB input"), *[(r, REP_LABEL[r]) for r in REPS]]
    keys, cols = [k for k, _ in columns], [lab for _, lab in columns]
    fig, axes = plt.subplots(len(rows), len(keys), figsize=(13 * len(keys) / 4, 3.3 * len(rows)), facecolor="white")
    for r, (ds, views) in enumerate(rows):
        for c, key in enumerate(keys):
            ax = axes[r, c]
            im = views[key]
            if key == "rgb":
                ax.imshow(im)
            else:
                lo, hi = ranges[key]
                ax.imshow(np.clip((im - lo) / (hi - lo), 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=12, color=INK, pad=8)
            if c == 0:
                ax.set_ylabel(f"{ds}\n{views['id']}", fontsize=11, color=INK)
    if title is None:
        title = "Zoomed vessel region (same physical size in every dataset)" if zoom else "What each input looks like to the network"
        title += " — grey levels use one shared scale per column, so brightness differences between rows are real"
    fig.suptitle(title, fontsize=11, color=MUTED, y=0.995)
    if caption:
        wrapped = "\n".join(textwrap.fill(p, int(fig.get_figwidth() * 72 / (9 * 0.62))) for p in caption.split("\n"))
        n_lines = wrapped.count("\n") + 1
        bottom = min(0.3, (0.16 * n_lines + 0.3) / fig.get_figheight())
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.text(0.01, bottom - 0.01, wrapped, ha="left", va="top", fontsize=9, color=INK, family="DejaVu Sans")
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def final_caption(ckpts, ranges_note):
    """Caption for the final-model gallery and zoom (also written to enhancement_final_caption.md)."""
    lines = [
        "Columns: RGB input, green channel, CLAHE (green), hybrid (per-patch min-max), hybrid_weber (multi-scale Weber "
        "normalization, final Part 1 model). Grey scales differ between columns, so only comparisons within a row are "
        f"meaningful: each column has one shared grey scale ({ranges_note}), and each row uses its own checkpoint. "
        "Both enhanced columns in a row come from the same fold and seed.",
        "Checkpoints (seed 1): " + "; ".join(
            f"{row} {path}" for row, path in ckpts.items()) + ".",
        "Seed 1 is the hybrid_weber seed with the low HRF result (LODO HRF Dice 0.760 against 0.809 for hybrid, "
        "-0.049; seeds 2 and 3 do not repeat it), so the HRF row shows the weakest hybrid_weber seed on that dataset.",
    ]
    return "\n".join(lines)


def run_final(args, device):
    """Gallery + zoom: RGB | green | CLAHE | hybrid | hybrid_weber, one row per dataset plus FIVES (seed 1)."""
    out = Path(args.out or "figures/enhancement_final")
    out.mkdir(parents=True, exist_ok=True)
    runs = Path(args.runs)
    rng = np.random.default_rng(0)
    samples = {k: [] for k, _ in FINAL_COLS if k != "rgb"}
    views_by_row, ckpts, record = [], {}, []

    for label, pool, img_id, split, fold in FINAL_ROWS:
        pool = Path(pool)
        meta = load_metadata(pool)[img_id]
        models, data_cfgs = {}, {}
        for v in FINAL_VARIANTS:
            path = runs / v / split / fold / f"seed{FINAL_SEED}" / "best.pt"
            if not path.exists():
                raise SystemExit(f"missing checkpoint {path}")
            models[v], data_cfgs[v] = load_variant(v, path, device)
            record.append({"row": label, "image": img_id, "variant": v, "checkpoint": str(path)})
        ckpts[f"{label} row"] = f"{runs}/{{{','.join(FINAL_VARIANTS)}}}/{split}/{fold}/seed{FINAL_SEED}/best.pt"
        # both models must see the image the same way, or the two columns are not comparable
        prep = {v: (working_scale(meta, c), bool(c.get("fov_norm", False)), bool(c.get("mask_fov", False)))
                for v, c in data_cfgs.items()}
        if len(set(prep.values())) != 1:
            raise SystemExit(f"{label}: hybrid and hybrid_weber use different input preprocessing {prep}")
        scale, fov_norm, mask_fov = prep[FINAL_VARIANTS[0]]

        rgb = read_rgb(pool / "images" / f"{img_id}.png")
        gt = read_binary(pool / "masks" / f"{img_id}.png")
        fov = read_binary(pool / "fov" / f"{img_id}.png")
        rgb, gt, fov = resize_sample(rgb, gt, fov, scale)
        model_rgb = fov_normalize(rgb, fov) if fov_norm else (fov_mask_image(rgb, fov) if mask_fov else rgb)
        views = {"green": rgb[..., 1].astype(np.float32) / 255.0,
                 "clahe": preprocess(rgb, "clahe")[..., 0].astype(np.float32) / 255.0}
        for v in FINAL_VARIANTS:
            views[v] = enhance_full(models[v], model_rgb, device, fov=fov)
        for k in samples:
            samples[k].append(rng.choice(views[k][fov > 0], 4000))
        views_by_row.append((label, img_id, meta, scale, rgb, gt, fov, views))
        print(f"{label:9} {img_id:18} " + "  ".join(
            f"{k} CNR {contrast_stats(views[k], gt, fov)['cnr']:.2f}" for k in samples))
        del models

    # one display range per column, shared by all rows (1st-99th percentile of FOV pixels over the five rows)
    ranges = {k: tuple(np.percentile(np.concatenate(s), [1, 99])) for k, s in samples.items()}
    caption = final_caption(ckpts, "1st-99th percentile of FOV pixels over all five rows")

    full_rows, zoom_rows = [], []
    for label, img_id, meta, scale, rgb, gt, fov, views in views_by_row:
        f = 900 / max(rgb.shape[:2])
        small = lambda a: cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else a
        full_rows.append((label, {"id": img_id, "rgb": small(rgb), **{k: small(views[k]) for k in samples}}))
        # zoom: same rows, columns, scales and caption as the gallery; window = 22% of the eye diameter
        win = max(64, int(0.22 * int(meta["fov_diameter"]) * scale))
        cy, cx = vessel_dense_centre(gt, fov, win)
        y0, x0 = max(0, cy - win // 2), max(0, cx - win // 2)
        crop = lambda a: cv2.resize(a[y0:y0 + win, x0:x0 + win], (360, 360),
                                    interpolation=cv2.INTER_AREA if win > 360 else cv2.INTER_NEAREST)
        zoom_rows.append((label, {"id": img_id, "rgb": crop(rgb), **{k: crop(views[k]) for k in samples}}))

    gallery(full_rows, ranges, out / "enhancement_final_gallery.png", columns=FINAL_COLS, caption=caption,
            title="Original module (hybrid) vs final model (hybrid_weber), seed 1")
    gallery(zoom_rows, ranges, out / "enhancement_final_zoom.png", zoom=True, columns=FINAL_COLS, caption=caption,
            title="Zoomed vessel region (22% of the eye diameter): hybrid vs hybrid_weber, seed 1")
    (out / "enhancement_final_caption.md").write_text(caption + "\n")
    with open(out / "enhancement_final_checkpoints.json", "w") as fh:
        json.dump({"seed": FINAL_SEED, "display_ranges": {k: [float(a), float(b)] for k, (a, b) in ranges.items()},
                   "rows": record}, fh, indent=2)
    print(f"figures written to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="single-checkpoint mode: the model to visualize")
    ap.add_argument("--final", action="store_true",
                    help="compare hybrid and hybrid_weber (seed 1) per dataset plus FIVES; ignores --checkpoint")
    ap.add_argument("--runs", default="runs", help="--final: root of the run folders")
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--out", default=None,
                    help="output folder (default figures/enhancement_run1, or figures/enhancement_final with --final)")
    ap.add_argument("--device", default="auto",
                    help="auto (cuda, then mps, then cpu), or e.g. cpu / cuda:1; use cpu to leave a shared GPU alone")
    ap.add_argument("--ids", nargs="*", help="limit to these image ids (default: all pooled images)")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device}")
    if args.final:
        run_final(args, device)
        return
    if not args.checkpoint:
        ap.error("--checkpoint is required unless --final is given")
    out = Path(args.out or "figures/enhancement_run1")
    out.mkdir(parents=True, exist_ok=True)
    model, fov_norm, data_cfg = load_model(args.checkpoint, device)
    if hasattr(model.enhance, "describe"):  # EnhancementV2
        learned = model.enhance.describe()
        print(f"learned enhancement parameters: {learned}  (initial: mix [0,1,0], sigma 48 px, gain 4)")
    else:
        c = model.enhance.contrast
        learned = {"gamma": c.gamma.item(), "alpha": c.alpha.item(), "beta": c.beta.item()}
        print(f"learned contrast parameters: {learned}  (initial values 1, 1, 0)")

    pool = Path(args.pool)
    meta = load_metadata(pool)
    ids = args.ids or sorted(meta)
    rng = np.random.default_rng(0)
    samples = {k: [] for k in REPS}        # pixel samples for shared display ranges / histograms
    bg_by_ds = {k: {d: [] for d in DATASETS} for k in REPS}
    rows, showcase = [], {}

    for n, i in enumerate(ids):
        ds = meta[i]["dataset"]
        rgb = read_rgb(pool / "images" / f"{i}.png")
        gt = read_binary(pool / "masks" / f"{i}.png")
        fov = read_binary(pool / "fov" / f"{i}.png")
        # show the image at the scale the model was trained on (scale normalization), as the network sees it
        rgb, gt, fov = resize_sample(rgb, gt, fov, working_scale(meta[i], data_cfg))
        reps = representations(model, rgb, fov, device, fov_norm, mask_fov=bool(data_cfg.get("mask_fov", False)))
        row = {"id": i, "dataset": ds}
        for k, img in reps.items():
            st = contrast_stats(img, gt, fov)
            row.update({f"{k}_{m}": round(v, 5) for m, v in st.items()})
            pix = img[fov > 0]
            samples[k].append(rng.choice(pix, 4000))
            bgpix = img[(fov > 0) & (gt == 0)]
            bg_by_ds[k][ds].append(rng.choice(bgpix, 4000))
        rows.append(row)
        if SHOWCASE.get(ds) == i:
            showcase[ds] = (rgb, gt, fov, reps)
        print(f"[{n + 1}/{len(ids)}] {ds:9} {i:14} CNR green {row['green_cnr']:.2f}  clahe {row['clahe_cnr']:.2f}  "
              f"enhanced {row['enhanced_cnr']:.2f}")

    with open(out / "enhancement_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # one display range per representation, shared by all datasets (1st-99th percentile of FOV pixels)
    ranges = {k: tuple(np.percentile(np.concatenate(samples[k]), [1, 99])) for k in REPS}

    # --- dataset gap: how far apart the datasets' typical background intensity is, relative to the overall spread
    summary = {"checkpoint": args.checkpoint, "learned_contrast": learned, "n_images": len(rows), "representations": {}}
    for k in REPS:
        allpix = np.concatenate(samples[k])
        iqr = float(np.subtract(*np.percentile(allpix, [75, 25])))
        med = {d: float(np.median([r[f"{k}_background_median"] for r in rows if r["dataset"] == d]))
               for d in DATASETS if any(r["dataset"] == d for r in rows)}
        vals = list(med.values())
        gap = float(np.mean([abs(a - b) for x, a in enumerate(vals) for b in vals[x + 1:]]) / iqr) if len(vals) > 1 else 0.0
        cnr = {d: float(np.mean([r[f"{k}_cnr"] for r in rows if r["dataset"] == d])) for d in med}
        summary["representations"][k] = {
            "dataset_gap": round(gap, 3),
            "cnr_by_dataset": {d: round(v, 3) for d, v in cnr.items()},
            "cnr_mean": round(float(np.mean(list(cnr.values()))), 3),
            "cnr_spread_between_datasets": round(float(np.std(list(cnr.values())) / np.mean(list(cnr.values()))), 3),
        }
    save_json(summary, out / "enhancement_summary.json")

    # --- figure 1 & 2: gallery and zoom
    full_rows, zoom_rows = [], []
    for ds in DATASETS:
        if ds not in showcase:
            continue
        rgb, gt, fov, reps = showcase[ds]
        scale = 900 / max(rgb.shape[:2])
        small = lambda a: cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else a
        full_rows.append((ds, {"id": SHOWCASE[ds], "rgb": small(rgb), **{k: small(v) for k, v in reps.items()}}))
        win = max(64, int(0.22 * int(meta[SHOWCASE[ds]]["fov_diameter"]) * working_scale(meta[SHOWCASE[ds]], data_cfg)))
        cy, cx = vessel_dense_centre(gt, fov, win)
        y0, x0 = max(0, cy - win // 2), max(0, cx - win // 2)
        crop = lambda a: cv2.resize(a[y0:y0 + win, x0:x0 + win], (360, 360), interpolation=cv2.INTER_AREA
                                    if win > 360 else cv2.INTER_NEAREST)
        zoom_rows.append((ds, {"id": SHOWCASE[ds], "rgb": crop(rgb), **{k: crop(v) for k, v in reps.items()}}))
    if full_rows:
        gallery(full_rows, ranges, out / "enhancement_gallery.png")
        gallery(zoom_rows, ranges, out / "enhancement_zoom.png", zoom=True)

    # --- figure 3: background intensity distributions per dataset
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor="white")
    for ax, k in zip(axes, REPS):
        lo, hi = np.percentile(np.concatenate(samples[k]), [1, 99.5])
        if k == "enhanced":
            bins = np.linspace(lo, hi, 70)
        else:  # 8-bit inputs: align bin edges to the 1/255 grid, otherwise the curves look comb-like
            step = 3 / 255
            bins = np.arange(np.floor(lo * 255) / 255 - 0.5 / 255, hi + step, step)
        for d in DATASETS:
            if not bg_by_ds[k][d]:
                continue
            h, e = np.histogram(np.concatenate(bg_by_ds[k][d]), bins=bins, density=True)
            xs = (e[:-1] + e[1:]) / 2
            ax.plot(xs, h, color=COLORS[d], linewidth=2, label=d)
            ax.annotate(d, (xs[np.argmax(h)], h.max()), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
        style(ax)
        ax.set_yticks([])
        ax.set_title(f"{REP_LABEL[k]}\ndataset gap {summary['representations'][k]['dataset_gap']:.2f}",
                     fontsize=11, color=INK)
        ax.set_xlabel("background intensity inside the FOV", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.suptitle("Do the four datasets look alike to the network?  Lower dataset gap = distributions closer together "
                 "(median distance between datasets ÷ overall spread)", fontsize=10.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "enhancement_histograms.png", dpi=130)
    plt.close(fig)

    # --- figure 4: vessel contrast-to-noise ratio, slope chart per dataset
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor="white")
    xs = np.arange(len(REPS))
    for d in DATASETS:
        ys = [summary["representations"][k]["cnr_by_dataset"].get(d) for k in REPS]
        if None in ys:
            continue
        ax.plot(xs, ys, color=COLORS[d], linewidth=2, marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=2, label=d)
        ax.annotate(f"{d}  {ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    style(ax)
    ax.set_xticks(xs, [REP_LABEL[k] for k in REPS], fontsize=10, color=INK)
    ax.set_xlim(-0.3, len(REPS) - 0.3)
    ax.set_ylabel("vessel contrast-to-noise ratio", fontsize=10, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.set_title("How strongly vessels stand out from the background (higher = more visible)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out / "enhancement_contrast.png", dpi=130)
    plt.close(fig)

    print("\nsummary:")
    for k in REPS:
        s = summary["representations"][k]
        print(f"  {REP_LABEL[k]:20} dataset gap {s['dataset_gap']:.3f}   mean CNR {s['cnr_mean']:.2f}   "
              f"CNR spread between datasets {s['cnr_spread_between_datasets']:.2f}")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
