"""Does the enhancement act like an unsharp mask? Average intensity profile across vessel edges.

For every image, each pixel inside the FOV gets its signed distance to the nearest vessel edge (negative = inside the
vessel). Each representation is expressed in background-noise units (mean 0, sd 1 over pixels 8-15 px away from any
vessel), then averaged per distance. An unsharp mask shows two signatures: a deeper vessel and a bright "halo"
(overshoot above the background) just outside the edge.

Usage:
    python edge_profile.py --checkpoint runs_check/enh2c_scale_strong/lodo/lodo_STARE/seed1/best.pt
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from u2nete.data import fov_mask_image, fov_normalize, load_metadata, preprocess, resize_sample, working_scale
from u2nete.io import read_binary, read_rgb
from u2nete.utils import get_device, save_json
from visualize_enhancement import GRID, INK, MUTED, load_model, style

BINS = np.array([-3, -2, -1, *range(1, 13)])  # distance transforms give no pixel at exactly 0
SERIES = [  # name, colour (validated palette, fixed order), dash: secondary encoding for colour-vision deficiency
    ("Green channel", "#2a78d6", (0, ())),
    ("Unsharp mask (σ 2 px, amount 1.5)", "#eb6834", (0, (6, 2))),
    ("CLAHE (green)", "#1baf7a", (0, (2, 2))),
    ("V2 without residual CNN", "#eda100", (0, (8, 2, 2, 2))),
    ("V2 full", "#e87ba4", (0, ())),
]


def unsharp(img, sigma=2.0, amount=1.5):
    return img + amount * (img - cv2.GaussianBlur(img, (0, 0), sigma))


@torch.no_grad()
def v2_maps(model, rgb, device, patch=384, fov=None):
    e = model.enhance
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None].to(device) / 255.0
    kw = {"fov": torch.from_numpy(fov.astype(np.float32))[None, None].to(device)} \
        if (fov is not None and getattr(e, "accepts_fov", False)) else {}
    full = e.enhance_image(x, patch, **kw)[0, 0].cpu().numpy()
    saved, e.residual = e.residual, None
    pre = e.enhance_image(x, patch, **kw)[0, 0].cpu().numpy()
    e.residual = saved
    return pre, full


def profile(img, signed, fov):
    bg = (signed >= 8) & (signed <= 15) & fov
    z = (img - img[bg].mean()) / (img[bg].std() + 1e-8)
    d = np.round(signed).astype(int)
    return np.array([z[(d == b) & fov].mean() if ((d == b) & fov).any() else np.nan for b in BINS])


@torch.no_grad()
def residual_probe(model, device):
    """Feed the residual CNN an ideal edge and an ideal thin vessel on a mid-grey background."""
    res = model.enhance.residual
    if res is None:
        return None
    w = 64
    step = np.full((1, 1, w, w), 0.5, np.float32)
    step[..., w // 2:] = 0.3                     # dark region right of the edge
    line = np.full((1, 1, w, w), 0.5, np.float32)
    line[..., w // 2 - 2:w // 2 + 2] = 0.3       # 4 px dark vessel
    out = {}
    for name, arr in (("edge", step), ("thin_vessel", line)):
        t = torch.from_numpy(arr).to(device)
        r = res(t)[0, 0, w // 2].cpu().numpy()
        o = (t + res(t)).clamp(0, 1)[0, 0, w // 2].cpu().numpy()
        out[name] = {"input": arr[0, 0, w // 2, w // 2 - 8:w // 2 + 8].round(3).tolist(),
                     "residual": r[w // 2 - 8:w // 2 + 8].round(3).tolist(),
                     "output": o[w // 2 - 8:w // 2 + 8].round(3).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--out", default="figures/edge_profile")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    model, fov_norm, data_cfg = load_model(args.checkpoint, device)
    if not hasattr(model.enhance, "enhance_image"):
        raise SystemExit("this script analyses EnhancementV2 checkpoints")
    pool = Path(args.pool)
    meta = load_metadata(pool)

    profiles = {name: [] for name, _, _ in SERIES}
    for n, i in enumerate(sorted(meta)):
        rgb = read_rgb(pool / "images" / f"{i}.png")
        gt = read_binary(pool / "masks" / f"{i}.png")
        fov = read_binary(pool / "fov" / f"{i}.png")
        rgb, gt, fov = resize_sample(rgb, gt, fov, working_scale(meta[i], data_cfg))
        mask_fov = bool(data_cfg.get("mask_fov", False))
        model_in = fov_normalize(rgb, fov) if fov_norm else (fov_mask_image(rgb, fov) if mask_fov else rgb)
        signed = distance_transform_edt(1 - gt) - distance_transform_edt(gt)
        f = fov > 0
        # vessels are dark in every representation; flip so "vessel depth" is positive and a halo is negative
        green = rgb[..., 1].astype(np.float32) / 255
        pre, full = v2_maps(model, model_in, device, fov=fov)
        reps = {
            "Green channel": green,
            "Unsharp mask (σ 2 px, amount 1.5)": unsharp(green),
            "CLAHE (green)": preprocess(rgb, "clahe")[..., 0].astype(np.float32) / 255,
            "V2 without residual CNN": pre,
            "V2 full": full,
        }
        for name, img in reps.items():
            profiles[name].append(profile(img, signed, f))
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(meta)} images")

    mean = {k: np.nanmean(np.stack(v), axis=0) for k, v in profiles.items()}
    centre = BINS <= -1
    outside = (BINS >= 1) & (BINS <= 5)
    summary = {}
    for k, p in mean.items():
        summary[k] = {
            "vessel_depth": round(float(-np.nanmin(p[centre])), 2),             # how dark the vessel core is
            "edge_slope": round(float(np.nanmax(np.diff(p))), 2),              # steepest change across the edge
            "halo": round(float(max(0.0, np.nanmax(p[outside]))), 2),          # bright overshoot just outside
        }
    probe = residual_probe(model, device)
    save_json({"profile_bins_px": BINS.tolist(), "profiles": {k: np.round(v, 3).tolist() for k, v in mean.items()},
               "summary": summary, "residual_probe": probe, "checkpoint": args.checkpoint},
              out / "edge_profile.json")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    for name, color, dash in SERIES:
        ax.plot(BINS, mean[name], color=color, linestyle=dash, linewidth=2, marker="o", markersize=4, label=name)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.axhline(0, color=GRID, linewidth=1)
    ax.text(0.15, ax.get_ylim()[1] * 0.95, "vessel edge", fontsize=9, color=MUTED, va="top")
    style(ax)
    ax.set_xlim(BINS[0] - 0.5, BINS[-1] + 0.5)
    ax.set_xticks(BINS)
    ax.set_xlabel("distance from vessel edge (px at training scale; negative = inside the vessel)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("intensity in background-noise units", fontsize=9.5, color=MUTED)
    ax.set_title("Average profile across vessel edges (all 113 images): deeper dip = more vessel contrast,\n"
                 "steeper wall = sharper edge, bump above 0 just outside = halo", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "edge_profile.png", dpi=140)
    plt.close(fig)

    print("\n{:36} {:>12} {:>10} {:>6}".format("", "vessel depth", "edge slope", "halo"))
    for k, s in summary.items():
        print(f"{k:36} {s['vessel_depth']:12.2f} {s['edge_slope']:10.2f} {s['halo']:6.2f}")
    if probe:
        print("\nresidual CNN on an ideal edge (centre row, 16 px around the edge):")
        print("  input   ", probe["edge"]["input"])
        print("  residual", probe["edge"]["residual"])
        print("residual CNN on an ideal 4 px vessel:")
        print("  input   ", probe["thin_vessel"]["input"])
        print("  residual", probe["thin_vessel"]["residual"])


if __name__ == "__main__":
    main()
