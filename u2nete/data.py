"""Loading the pooled dataset, scale normalization, preprocessing and patch sampling."""
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import PHOTOMETRIC, geometric, photometric_shift
from .io import read_binary, read_rgb


@dataclass
class Sample:
    id: str
    dataset: str
    image: np.ndarray      # uint8 RGB at working scale
    gt: np.ndarray         # uint8 {0,1} at working scale
    fov: np.ndarray        # uint8 {0,1} at working scale
    scale: float           # working / original
    orig_shape: tuple      # (H, W) of the original image


def load_metadata(pool_dir):
    with open(Path(pool_dir) / "metadata.csv") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def resize_sample(img, gt, fov, scale):
    if abs(scale - 1.0) < 1e-3:
        return img, gt, fov
    h, w = gt.shape
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, size, interpolation=interp)
    gt = (cv2.resize(gt.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    fov = (cv2.resize(fov.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    return img, gt, fov


def working_scale(meta_row, data_cfg) -> float:
    """Scale factor that brings every eye to the same FOV diameter (if scale_norm is on)."""
    if not data_cfg.get("scale_norm", False):
        return 1.0
    return float(data_cfg["target_fov"]) / float(meta_row["fov_diameter"])


def fov_normalize(img: np.ndarray, fov: np.ndarray, lo_pct=1.0, hi_pct=99.0) -> np.ndarray:
    """Stretch each channel to its 1st-99th percentile inside the FOV, once per image.

    Replaces the enhancement module's per-patch min-max, which makes the same pixel brighter or darker
    depending on which patch it falls in (visible as tiles in the enhanced image).
    """
    out = np.zeros_like(img)
    inside = fov > 0
    for c in range(3):
        v = img[..., c][inside]
        lo, hi = np.percentile(v, [lo_pct, hi_pct])
        ch = (img[..., c].astype(np.float32) - lo) / max(hi - lo, 1.0)
        out[..., c] = np.clip(ch * 255.0, 0, 255).astype(np.uint8)
    out[~inside] = 0
    return out


def fov_mask_image(img: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Black out everything outside the eye (FOV) mask without changing pixels inside it."""
    out = img.copy()
    out[fov == 0] = 0
    return out


def load_samples(ids, pool_dir, data_cfg, meta=None):
    return list(iter_samples(ids, pool_dir, data_cfg, meta))


def iter_samples(ids, pool_dir, data_cfg, meta=None):
    """Load samples one at a time (large external sets such as FIVES do not fit in memory at once)."""
    pool = Path(pool_dir)
    meta = meta or load_metadata(pool)
    for i in ids:
        img = read_rgb(pool / "images" / f"{i}.png")
        gt = read_binary(pool / "masks" / f"{i}.png")
        fov = read_binary(pool / "fov" / f"{i}.png")
        s = working_scale(meta[i], data_cfg)
        img_s, gt_s, fov_s = resize_sample(img, gt, fov, s)
        if data_cfg.get("fov_norm", False):
            img_s = fov_normalize(img_s, fov_s)
        elif data_cfg.get("mask_fov", False):
            img_s = fov_mask_image(img_s, fov_s)
        yield Sample(i, meta[i]["dataset"], img_s, gt_s, fov_s, s, gt.shape)


def preprocess(img: np.ndarray, mode: str) -> np.ndarray:
    """Fixed (non-learned) input preprocessing. Returns uint8 RGB."""
    if mode == "rgb":
        return img
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(np.ascontiguousarray(img[..., 1]))
        return np.stack([g, g, g], axis=-1)
    raise ValueError(f"unknown preprocess mode {mode}")


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0


class PatchDataset(Dataset):
    """Infinite stream of random training patches.

    Datasets are sampled with equal probability (domain-balanced), so HRF's 36 large images don't dominate,
    and patch centres are drawn from inside the FOV.
    """

    def __init__(self, samples, cfg, length=10_000_000, seed=0):
        self.samples = samples
        self.patch = int(cfg["data"]["patch_size"])
        self.preproc = cfg["data"]["preprocess"]
        aug = cfg["augment"]
        self.photometric = PHOTOMETRIC[aug["photometric"]]
        self.rotate_deg = float(aug.get("rotate_deg", 20))
        self.scale_range = tuple(aug["scale_range"]) if aug.get("scale_range") else None
        self.full_image_prob = float(cfg["data"].get("full_image_prob", 0.0))
        self.balance = cfg["data"].get("balance_datasets", True)
        self.mask_fov = bool(cfg["data"].get("mask_fov", False) or cfg["data"].get("fov_norm", False))
        self.paired = float(cfg["loss"].get("consistency_weight", 0.0)) > 0
        self.length = length
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

        by_ds = {}
        for k, s in enumerate(self.samples):
            by_ds.setdefault(s.dataset, []).append(k)
        self.by_dataset = list(by_ds.values())
        # pre-compute FOV pixel coordinates (subsampled) for centre sampling
        self.centres = []
        for s in self.samples:
            ys, xs = np.nonzero(s.fov[::4, ::4])
            self.centres.append(np.stack([ys * 4, xs * 4], 1))

    def worker_init(self, worker_id):
        info = torch.utils.data.get_worker_info()
        self.rng = np.random.default_rng([self.base_seed, worker_id, info.seed % (2**31) if info else 0])

    def __len__(self):
        return self.length

    def _pick(self):
        if self.balance:
            group = self.by_dataset[self.rng.integers(len(self.by_dataset))]
            return group[self.rng.integers(len(group))]
        return int(self.rng.integers(len(self.samples)))

    def __getitem__(self, _):
        rng = self.rng
        k = self._pick()
        s = self.samples[k]
        p = self.patch

        if rng.random() < self.full_image_prob:
            # whole image downscaled to one patch (as in the original run)
            img = cv2.resize(s.image, (p, p), interpolation=cv2.INTER_AREA)
            gt = (cv2.resize(s.gt.astype(np.float32), (p, p)) > 0.5).astype(np.uint8)
            fov = (cv2.resize(s.fov.astype(np.float32), (p, p)) > 0.5).astype(np.uint8)
            img, gt, fov = geometric(img, gt, fov, rng, 0, None)
        else:
            # crop with a margin so rotation/scale don't introduce empty borders, then centre-crop
            m = int(p * 0.75)
            cy, cx = self.centres[k][rng.integers(len(self.centres[k]))]
            img, gt, fov = (self._crop(a, int(cy), int(cx), m) for a in (s.image, s.gt, s.fov))
            img, gt, fov = geometric(img, gt, fov, rng, self.rotate_deg, self.scale_range)
            o = (img.shape[0] - p) // 2
            img, gt, fov = img[o:o + p, o:o + p], gt[o:o + p, o:o + p], fov[o:o + p, o:o + p]

        img = self.photometric(np.ascontiguousarray(img), rng)
        if self.mask_fov:
            img[fov == 0] = 0
        gt_t = torch.from_numpy(gt.astype(np.float32))[None]
        fov_t = torch.from_numpy(fov.astype(np.float32))[None]
        if self.paired:
            # second view of the same patch for the enhancement-consistency loss
            img_b = photometric_shift(img, rng)
            if self.mask_fov:
                img_b[fov == 0] = 0
            return (to_tensor(preprocess(img, self.preproc)), gt_t, fov_t,
                    to_tensor(preprocess(img_b, self.preproc)))
        return to_tensor(preprocess(img, self.preproc)), gt_t, fov_t

    @staticmethod
    def _crop(a, cy, cx, half):
        """Crop a (2*half)x(2*half) window centred at (cy, cx), reflect-padding at the borders."""
        pad = [(half, half), (half, half)] + [(0, 0)] * (a.ndim - 2)
        a = np.pad(a, pad, mode="reflect")
        return a[cy:cy + 2 * half, cx:cx + 2 * half]
