"""Full-image inference: sliding window with Gaussian blending and optional 8-fold TTA.

Unlike the original test.py there is no per-image min-max rescaling of the output: the sigmoid output is
already a probability, and rescaling changes what the 0.5 threshold means from image to image.
"""
import cv2
import numpy as np
import torch

from .data import fov_mask_image, preprocess, to_tensor
from .model import prediction


def _gaussian(p, sigma_frac=0.125):
    ax = np.arange(p) - (p - 1) / 2
    g = np.exp(-(ax ** 2) / (2 * (sigma_frac * p) ** 2))
    w = np.outer(g, g)
    return (w / w.max()).astype(np.float32) + 1e-3


def backbone(model):
    """The network after the enhancement front-end (applies the optional ImageNet normalization)."""
    def run(x):
        if model.normalize:
            x = (x - model.mean) / model.std
        return model.u2net(x)
    return run


@torch.no_grad()
def sliding_window(model, img_t, patch, overlap, batch, device, which, net=None):
    """img_t: (3,H,W) tensor. Returns (H,W) numpy probability map. `net` defaults to the whole model."""
    net = net or model
    _, H, W = img_t.shape
    ph, pw = max(0, patch - H), max(0, patch - W)
    if ph or pw:
        img_t = torch.nn.functional.pad(img_t[None], (0, pw, 0, ph), mode="reflect")[0]
    _, Hp, Wp = img_t.shape
    stride = patch - overlap
    ys = list(range(0, Hp - patch + 1, stride))
    xs = list(range(0, Wp - patch + 1, stride))
    if ys[-1] != Hp - patch:
        ys.append(Hp - patch)
    if xs[-1] != Wp - patch:
        xs.append(Wp - patch)
    coords = [(y, x) for y in ys for x in xs]

    w = torch.from_numpy(_gaussian(patch)).to(device)
    acc = torch.zeros((Hp, Wp), device=device)
    norm = torch.zeros((Hp, Wp), device=device)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        x = torch.stack([img_t[:, y:y + patch, x:x + patch] for y, x in chunk]).to(device)
        p = prediction(net(x), which)[:, 0]
        for (y, x0), pi in zip(chunk, p):
            acc[y:y + patch, x0:x0 + patch] += pi * w
            norm[y:y + patch, x0:x0 + patch] += w
    return (acc / norm)[:H, :W].float().cpu().numpy()


_TTA = [
    (lambda a: a, lambda a: a),
    (lambda a: np.flip(a, 1), lambda a: np.flip(a, 1)),
    (lambda a: np.flip(a, 0), lambda a: np.flip(a, 0)),
    (lambda a: np.rot90(a, 1), lambda a: np.rot90(a, -1)),
    (lambda a: np.rot90(a, 2), lambda a: np.rot90(a, -2)),
    (lambda a: np.rot90(a, 3), lambda a: np.rot90(a, -3)),
    (lambda a: np.rot90(np.flip(a, 1), 1), lambda a: np.flip(np.rot90(a, -1), 1)),
    (lambda a: np.rot90(np.flip(a, 1), 3), lambda a: np.flip(np.rot90(a, -3), 1)),
]


def predict_sample(model, sample, cfg, device, tta=False, to_original=True):
    """Probability map for a Sample; resized back to the original resolution if to_original."""
    inf = cfg["inference"]
    img = preprocess(sample.image, cfg["data"]["preprocess"])
    if cfg["data"].get("mask_fov", False):
        img = fov_mask_image(img, sample.fov)
    transforms = _TTA if tta else _TTA[:1]
    probs = []
    whole_image = hasattr(model.enhance, "enhance_image") and (
        getattr(model.enhance, "supports_any_size", False) or img.shape[0] * img.shape[1] <= 6_000_000
    )
    accepts_fov = getattr(model.enhance, "accepts_fov", False)
    for fwd, inv in transforms:
        t = to_tensor(np.ascontiguousarray(fwd(img)))
        if whole_image:
            # EnhancementV2/V3: enhance the whole image once (no patch seams), then slide the backbone over it
            if accepts_fov:
                fov_t = torch.from_numpy(np.ascontiguousarray(fwd(sample.fov))).float()[None, None].to(device)
                enh = model.enhance.enhance_image(t[None].to(device), int(cfg["data"]["patch_size"]), fov=fov_t)[0]
            else:
                enh = model.enhance.enhance_image(t[None].to(device), int(cfg["data"]["patch_size"]))[0]
            p = sliding_window(model, enh, int(cfg["data"]["patch_size"]), int(inf["overlap"]),
                               int(inf["batch"]), device, cfg["model"]["output"], net=backbone(model))
        else:
            p = sliding_window(model, t, int(cfg["data"]["patch_size"]), int(inf["overlap"]),
                               int(inf["batch"]), device, cfg["model"]["output"])
        probs.append(np.ascontiguousarray(inv(p)))
        if device.type == "cuda":
            # free the whole-image buffers of this pass before the next one (shared GPU)
            enh = t = None
            torch.cuda.empty_cache()
    prob = np.mean(probs, axis=0)
    if to_original and prob.shape != tuple(sample.orig_shape):
        H, W = sample.orig_shape
        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    return prob
