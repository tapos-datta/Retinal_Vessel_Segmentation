"""Geometric and photometric augmentation on uint8 patches.

"basic" mirrors the original run (flip/rotate/brightness/blur/noise).
"strong" adds domain randomization: colour, contrast, gamma, uneven illumination, blur, noise, JPEG and
resolution loss. These are the acquisition differences between fundus cameras, so randomizing them during
training is the main tool for generalizing to an unseen dataset.
"""
import cv2
import numpy as np


def geometric(img, gt, fov, rng, rotate_deg=20.0, scale_range=(0.8, 1.25)):
    """img/gt/fov are already cropped with a margin; returns arrays of the same size."""
    if rng.random() < 0.5:
        img, gt, fov = img[:, ::-1], gt[:, ::-1], fov[:, ::-1]
    if rng.random() < 0.5:
        img, gt, fov = img[::-1], gt[::-1], fov[::-1]
    k = int(rng.integers(4))
    if k:
        img, gt, fov = np.rot90(img, k), np.rot90(gt, k), np.rot90(fov, k)
    img, gt, fov = (np.ascontiguousarray(a) for a in (img, gt, fov))

    angle = float(rng.uniform(-rotate_deg, rotate_deg)) if rotate_deg else 0.0
    scale = float(np.exp(rng.uniform(np.log(scale_range[0]), np.log(scale_range[1])))) if scale_range else 1.0
    if angle or scale != 1.0:
        h, w = gt.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # labels: linear then threshold keeps thin vessels better than nearest-neighbour
        gt = (cv2.warpAffine(gt.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101) > 0.5).astype(np.uint8)
        fov = (cv2.warpAffine(fov.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101) > 0.5).astype(np.uint8)
    return img, gt, fov


def photometric_basic(img, rng):
    out = img.astype(np.float32)
    out *= 1.0 + 0.2 * (rng.random() - 0.5)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < 0.5:
        out += rng.normal(0, 5, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def photometric_strong(img, rng):
    out = img.astype(np.float32)
    h, w = out.shape[:2]

    # colour: per-channel gain, hue and saturation (camera white balance / sensor differences)
    out *= rng.uniform(0.8, 1.2, size=3).astype(np.float32)
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
        hsv[..., 1] *= rng.uniform(0.7, 1.3)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    # contrast / brightness / gamma (exposure)
    out = out * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    out = np.clip(out, 0, 255)
    out = 255.0 * (out / 255.0) ** rng.uniform(0.7, 1.5)

    # uneven illumination: smooth linear gradient across the patch
    if rng.random() < 0.5:
        theta = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ramp = (np.cos(theta) * (xx / w - 0.5) + np.sin(theta) * (yy / h - 0.5))
        out *= (1.0 + rng.uniform(-0.4, 0.4) * ramp)[..., None]

    # optics and sensor: blur, resolution loss, noise, compression
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.2:
        f = rng.uniform(0.5, 0.9)
        small = cv2.resize(out, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:
        out += rng.normal(0, rng.uniform(1, 8), out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 95))])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return out


def photometric_shift(img, rng):
    """Colour, exposure and illumination changes only: what an enhancement step should be able to cancel.

    Used to make the second view for the consistency loss (no blur/noise/JPEG, which no enhancement can undo).
    """
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    out *= rng.uniform(0.75, 1.25, size=3).astype(np.float32)
    out = out * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
    out = 255.0 * (np.clip(out, 0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
    theta = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = np.cos(theta) * (xx / w - 0.5) + np.sin(theta) * (yy / h - 0.5)
    out *= (1.0 + rng.uniform(-0.4, 0.4) * ramp)[..., None]
    out[img.max(axis=-1) == 0] = 0  # keep the outside-FOV border black
    return np.clip(out, 0, 255).astype(np.uint8)


PHOTOMETRIC = {"none": lambda img, rng: img, "basic": photometric_basic, "strong": photometric_strong}
