import torch
import torch.nn.functional as F


def tversky(pred, target, mask, alpha, beta, eps=1e-6):
    """Tversky index per sample, computed inside `mask` (FOV)."""
    pred, target = pred * mask, target * mask
    dims = (1, 2, 3)
    tp = (pred * target).sum(dims)
    fp = (pred * (1 - target)).sum(dims)
    fn = ((1 - pred) * target).sum(dims)
    return (tp + eps) / (tp + alpha * fp + beta * fn + eps)


class DeepSupervisionLoss:
    """w_bce * BCE + w_tversky * (1 - Tversky), averaged over the fused and all side outputs.

    Matches the original run (0.5/0.5, alpha=0.3, beta=0.7) but restricted to the FOV when fov_only is set.
    """

    def __init__(self, cfg):
        self.w_bce = float(cfg.get("w_bce", 0.5))
        self.w_tv = float(cfg.get("w_tversky", 0.5))
        self.alpha = float(cfg.get("tversky_alpha", 0.3))
        self.beta = float(cfg.get("tversky_beta", 0.7))
        self.fov_only = bool(cfg.get("fov_only", True))

    def __call__(self, outputs, target, fov):
        mask = fov if self.fov_only else torch.ones_like(fov)
        total = 0.0
        for out in outputs:
            out = out.clamp(1e-6, 1 - 1e-6)
            bce = (F.binary_cross_entropy(out, target, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
            tv = 1 - tversky(out, target, mask, self.alpha, self.beta).mean()
            total = total + self.w_bce * bce + self.w_tv * tv
        return total / len(outputs)
