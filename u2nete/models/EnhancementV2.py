"""EnhancementV2: an image-adaptive, illumination-aware enhancement front-end.

Fixes the problems found in the original EnhancementModule:
  * no per-patch min-max stretch (it caused tile seams); the image is normalized once per image upstream
  * illumination is estimated at the right scale (a learnable Gaussian, ~5% of the eye diameter) and removed by
    division (Retinex), because uneven lighting multiplies brightness rather than adding to it
  * the tone curve is predicted per input from its content (Zero-DCE style), instead of 3 numbers shared by all images
  * a learnable colour mix (initialized to "green only") replaces the hard-coded green channel
  * every learned part starts as identity and the output is bounded, so it stays an interpretable enhanced image

Input: (B,3,H,W) RGB in [0,1], black outside the FOV. Output: (B,3,H,W), the enhanced grey image triplicated and
scaled to roughly [-1,1] for the backbone (same interface as EnhancementModule).
"""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedGaussianIllumination(nn.Module):
    """Smooth illumination estimate L via normalized (FOV-masked) Gaussian blur with a learnable sigma.

    Computed on a 4x downsampled map for speed; sigma is in pixels of the input.
    """

    def __init__(self, sigma=48.0, down=4):
        super().__init__()
        self.down = down
        self.sigma0 = float(sigma)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        self.radius = int(math.ceil(3 * 2 * sigma / down))  # covers sigma up to 2x the initial value

    def sigma(self):
        return self.log_sigma.exp().clamp(self.sigma0 / 4, self.sigma0 * 2)

    def _kernel(self, device, dtype):
        s = self.sigma() / self.down
        ax = torch.arange(-self.radius, self.radius + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (ax / s) ** 2)
        return k / k.sum()

    def _blur(self, x, k):
        r = self.radius
        x = F.pad(x, (r, r, 0, 0), mode="replicate")
        x = F.conv2d(x, k.view(1, 1, 1, -1))
        x = F.pad(x, (0, 0, r, r), mode="replicate")
        return F.conv2d(x, k.view(1, 1, -1, 1))

    def forward(self, y, mask):
        H, W = y.shape[-2:]
        ys = F.avg_pool2d(y * mask, self.down, ceil_mode=True)
        ms = F.avg_pool2d(mask, self.down, ceil_mode=True)
        k = self._kernel(y.device, y.dtype)
        L = self._blur(ys, k) / (self._blur(ms, k) + 1e-4)
        return F.interpolate(L, size=(H, W), mode="bilinear", align_corners=False)


class CurveEstimator(nn.Module):
    """Predicts per-input curve coefficients a_1..a_n in (-1,1) from a thumbnail (Zero-DCE style).

    The last layer is zero-initialized, so every curve starts as the identity.
    """

    def __init__(self, iters=4, width=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(width, iters)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, mask):
        t = F.adaptive_avg_pool2d(x * mask, 64)
        f = self.features(t)
        f = f.mean(dim=(2, 3))
        return torch.tanh(self.head(f))  # (B, iters)


class EnhancementV2(nn.Module):
    def __init__(self, illum_sigma=48.0, curve_iters=4, residual=True, retinex_gain=4.0):
        super().__init__()
        # colour mix, starts as "take the green channel"
        self.mix = nn.Conv2d(3, 1, 1)
        with torch.no_grad():
            self.mix.weight.copy_(torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1))
            self.mix.bias.zero_()
        self.illum = MaskedGaussianIllumination(illum_sigma)
        # maps log-reflectance (≈0 for background, <0 for vessels) into (0,1); learnable gain
        self.log_gain = nn.Parameter(torch.tensor(math.log(retinex_gain)))
        self.curve = CurveEstimator(curve_iters)
        self.residual = None
        if residual:
            self.residual = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        self.last = {}  # intermediate maps of the latest forward pass, for visualization

    @staticmethod
    def fov_mask(x):
        return (x.amax(dim=1, keepdim=True) > 0.02).float()

    def _pre_curve(self, x):
        x = x.float()
        mask = self.fov_mask(x)
        y = self.mix(x).clamp(min=0.0) + 1e-3                     # luminance-like, >0
        L = self.illum(y, mask).clamp(min=1e-3)                    # smooth illumination
        r = torch.log(y) - torch.log(L)                            # Retinex log-reflectance
        z = torch.sigmoid(self.log_gain.exp() * r)                 # (0,1); background ≈ 0.5, vessels lower
        return z, mask, L

    def _post_curve(self, z, a, mask):
        for i in range(a.shape[1]):
            ai = a[:, i].view(-1, 1, 1, 1)
            z = z + ai * z * (1 - z)                               # stays in [0,1] for a in [-1,1]
        if self.residual is not None:
            z = (z + self.residual(z)).clamp(0.0, 1.0)
        z = z * mask                                               # keep outside-FOV black
        return torch.cat([z, z, z], dim=1) * 2.0 - 1.0

    def forward(self, x):
        """Training path: works on patches; each patch gets its own curve.

        Always computed in float32 (log/divide steps lose accuracy in 16-bit; the module is tiny, so it costs nothing).
        """
        fp32 = torch.autocast(device_type="cuda", enabled=False) if x.is_cuda and torch.is_autocast_enabled() \
            else contextlib.nullcontext()
        with fp32:
            z, mask, L = self._pre_curve(x)
            a = self.curve(z, mask)
            self.last = {"illumination": L.detach(), "curve": a.detach()}
            return self._post_curve(z, a, mask)

    @torch.no_grad()
    def enhance_image(self, x, patch=384):
        """Inference path for one whole image (1,3,H,W): no patch seams.

        Illumination is estimated on the whole image, and one tone curve is used for the whole image: the mean of
        the curves the estimator predicts for the image's patches (the input size it was trained on).
        """
        z, mask, L = self._pre_curve(x)
        H, W = z.shape[-2:]
        tiles = []  # full-size tiles only: the size the estimator was trained on (and MPS pooling needs)
        for y0 in range(0, H - patch + 1, patch // 2):
            for x0 in range(0, W - patch + 1, patch // 2):
                m = mask[..., y0:y0 + patch, x0:x0 + patch]
                if m.mean() > 0.3:  # mostly inside the eye
                    tiles.append((z[..., y0:y0 + patch, x0:x0 + patch], m))
        if not tiles:  # image smaller than a patch, or no tile mostly inside the eye
            size = (patch, patch)
            tiles = [(F.interpolate(z, size=size, mode="bilinear", align_corners=False),
                      F.interpolate(mask, size=size, mode="nearest"))]
        a = torch.cat([self.curve(t, m) for t, m in tiles]).mean(0, keepdim=True)
        self.last = {"illumination": L, "curve": a}
        return self._post_curve(z, a, mask)

    def describe(self):
        w = self.mix.weight.detach().flatten().tolist()
        return {"colour_mix_rgb": [round(v, 3) for v in w], "illum_sigma_px": round(float(self.illum.sigma()), 1),
                "retinex_gain": round(float(self.log_gain.exp()), 3)}
