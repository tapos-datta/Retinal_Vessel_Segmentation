"""Segmentation metrics computed inside the FOV at the original image resolution (standard protocol)."""
import numpy as np
from skimage.morphology import skeletonize


def auc_roc(prob, gt, bins=2000):
    """ROC AUC from a probability histogram (accurate to ~1e-4, fast on 8M+ pixel HRF images)."""
    idx = np.clip((prob * (bins - 1)).round().astype(np.int64), 0, bins - 1)
    pos = np.bincount(idx[gt == 1], minlength=bins).astype(np.float64)
    neg = np.bincount(idx[gt == 0], minlength=bins).astype(np.float64)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    # P(score_pos > score_neg) + 0.5 P(tie)
    neg_below = np.cumsum(neg) - neg
    return float((pos * (neg_below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def cl_dice(pred, gt):
    """Centerline Dice (Shit et al., CVPR 2021): rewards connected vessel trees, not just pixel overlap."""
    sp, sg = skeletonize(pred.astype(bool)), skeletonize(gt.astype(bool))
    tprec = (sp & gt.astype(bool)).sum() / max(sp.sum(), 1)
    tsens = (sg & pred.astype(bool)).sum() / max(sg.sum(), 1)
    return float(2 * tprec * tsens / max(tprec + tsens, 1e-8))


def compute(prob, gt, fov, threshold=0.5, with_cldice=True):
    fov = fov.astype(bool)
    p = prob[fov]
    g = gt[fov].astype(np.uint8)
    b = (p >= threshold).astype(np.uint8)
    tp = float((b & g).sum())
    fp = float((b & (1 - g)).sum())
    fn = float(((1 - b) & g).sum())
    tn = float(((1 - b) & (1 - g)).sum())
    m = {
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "auc": auc_roc(p, g),
    }
    if with_cldice:
        pred_full = (prob >= threshold) & fov
        m["cldice"] = cl_dice(pred_full, gt.astype(bool) & fov)
    return m


METRIC_KEYS = ["dice", "cldice", "auc", "precision", "sensitivity", "specificity", "iou", "accuracy"]
