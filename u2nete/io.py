"""Image I/O, dataset identification and field-of-view (FOV) masks."""
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".ppm", ".gif", ".bmp"}


def read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:  # GIF and some TIFF variants
        img = np.array(Image.open(path).convert("RGB"))
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_binary(path, threshold: int = 127) -> np.ndarray:
    """Read a mask (any format, incl. GIF) as a {0,1} uint8 array."""
    arr = np.array(Image.open(path).convert("L"))
    return (arr > threshold).astype(np.uint8)


def write_png(path, arr) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), arr)


def identify(name: str) -> dict:
    """Infer dataset, patient group and condition from an original file name."""
    stem = Path(name).stem
    m = re.match(r"^(\d+)_(h|g|dr)$", stem)
    if m:
        cond = {"h": "healthy", "g": "glaucoma", "dr": "diabetic_retinopathy"}[m.group(2)]
        # HRF: each image is a different subject
        return {"id": stem, "dataset": "HRF", "group": f"HRF_{stem}", "condition": cond}
    m = re.match(r"^(\d+)_(training|test)$", stem)
    if m:
        return {"id": stem, "dataset": "DRIVE", "group": f"DRIVE_{stem}", "condition": ""}
    m = re.match(r"^Image_(\d+)([LR])$", stem)
    if m:
        # CHASE_DB1: left and right eye of the same child -> same group, never split apart
        return {"id": stem, "dataset": "CHASE_DB1", "group": f"CHASE_{m.group(1)}", "condition": ""}
    m = re.match(r"^im(\d+)$", stem)
    if m:
        return {"id": stem, "dataset": "STARE", "group": f"STARE_{stem}", "condition": ""}
    return {"id": stem, "dataset": "UNKNOWN", "group": stem, "condition": ""}


def auto_fov(rgb: np.ndarray, erode_px: int = 0) -> np.ndarray:
    """Estimate the circular fundus FOV from the red channel.

    Standard practice for datasets without official FOV masks (CHASE_DB1, STARE).
    """
    red = cv2.medianBlur(rgb[..., 0], 5).astype(np.float32)
    thr = max(10.0, 0.08 * float(np.percentile(red, 99)))
    fov = (red > thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fov, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fov = (labels == largest).astype(np.uint8)
    # fill holes (dark lesions inside the FOV)
    h, w = fov.shape
    flood = fov.copy() * 255
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 128)
    fov = ((flood != 128)).astype(np.uint8)
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    if erode_px > 0:
        fov = cv2.erode(fov, np.ones((2 * erode_px + 1,) * 2, np.uint8))
    return fov


def fov_diameter(fov: np.ndarray) -> int:
    ys, xs = np.where(fov > 0)
    if len(xs) == 0:
        return max(fov.shape)
    return int(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
