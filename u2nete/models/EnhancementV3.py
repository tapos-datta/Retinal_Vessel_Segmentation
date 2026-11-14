"""EnhancementV3: outlier-clipped multi-scale Weber normalization + chromatic/residual RGB fusion.

Designed to preserve U2-Net-E's cross-dataset LODO gains while fixing its failure modes on unseen dark/high-outlier
external data (FIVES Glaucoma/DR) and small-FOV images (DRIVE):

  1. Replaces unmasked, unclipped per-patch min-max stretching with FOV-masked, two-pass Winsorized (bright-outlier
     capped) multi-scale local Weber normalization with a learnable dark-noise floor:
       * bright optic discs / exudates (bright_outlier > 5x on FIVES Glaucoma) are soft-capped before illumination
         estimation and bounded via tanh, so they cannot bleed halos or crush faint vessel contrast
       * black FOV borders (min = 0 on DRIVE and peripheral patches) are excluded from local background estimation
       * a positive denominator noise floor prevents 15-25x amplification of 8-bit quantization noise in dark regions
       * purely local spatial operators match 384 px training patches to seamless whole-image inference at any scale

  2. Replaces the 1-channel [enh, enh, enh] bottleneck with 3-channel chromatic + gated context fusion ('tri_rg'):
       * Channel 0: primary enhanced vessel map (adaptive RGB mix -> Winsorized Weber -> multi-scale dilated residual)
       * Channel 1: Red-channel Winsorized Weber map (retains high-SNR vessel/disc structure when Green is attenuated
         in dark/cataractous eyes, while remaining exposure-invariant)
       * Channel 2: learnable gated blend of Channel 0 and FOV-masked raw intensity context (avoids discarding raw
         exposure/saturation cues needed on DRIVE and saturated optic discs)
"""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleMaskedIllumination(nn.Module):
    """FOV-masked multi-scale Gaussian background estimator.

    Computes normalized separable Gaussian blurs at multiple spatial scales (on pooled grids for speed) and blends
    them with learnable softmax weights so thin capillaries (DRIVE) and wide arcades (HRF, FIVES) are both handled
    at native resolution without resizing the eye.
    """

    def __init__(self, sigmas=(16.0, 48.0, 128.0), init_weights=(0.25, 0.50, 0.25)):
        super().__init__()
        assert len(sigmas) == len(init_weights)
        self.sigmas = tuple(float(s) for s in sigmas)
        self.downs = tuple(max(1, min(8, int(round(s / 8.0)))) for s in self.sigmas)
        self.radii = tuple(int(math.ceil(3.0 * s / d)) for s, d in zip(self.sigmas, self.downs))
        w = torch.tensor(init_weights, dtype=torch.float32)
        self.scale_logits = nn.Parameter((w / w.sum()).log())

    def weights(self):
        return F.softmax(self.scale_logits, dim=0)

    @staticmethod
    def _kernel(sigma_grid: float, radius: int, device, dtype):
        ax = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (ax / sigma_grid) ** 2)
        return k / k.sum()

    @staticmethod
    def _blur(x: torch.Tensor, k: torch.Tensor, radius: int) -> torch.Tensor:
        x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
        x = F.conv2d(x, k.view(1, 1, 1, -1))
        x = F.pad(x, (0, 0, radius, radius), mode="replicate")
        return F.conv2d(x, k.view(1, 1, -1, 1))

    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        H, W = y.shape[-2:]
        ym = y * mask
        weights = self.weights()
        acc = torch.zeros_like(y)
        for idx, (sigma, down, radius) in enumerate(zip(self.sigmas, self.downs, self.radii)):
            ys = F.avg_pool2d(ym, down, ceil_mode=True) if down > 1 else ym
            ms = F.avg_pool2d(mask, down, ceil_mode=True) if down > 1 else mask
            k = self._kernel(sigma / down, radius, y.device, y.dtype)
            Li = self._blur(ys, k, radius) / (self._blur(ms, k, radius) + 1e-4)
            if Li.shape[-2:] != (H, W):
                Li = F.interpolate(Li, size=(H, W), mode="bilinear", align_corners=False)
            acc = acc + weights[idx] * Li
        return acc


class DilatedResidualUnit(nn.Module):
    """Multi-scale residual unit combining 3x3 (dilation 1) and 3x3 (dilation 2) vessel filters."""

    def __init__(self, channels: int):
        super().__init__()
        half = channels // 2
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.local = nn.Conv2d(channels, half, 3, padding=1, dilation=1)
        self.dilated = nn.Conv2d(channels, channels - half, 3, padding=2, dilation=2)
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x), inplace=True)
        h = F.relu(torch.cat([self.local(h), self.dilated(h)], dim=1), inplace=True)
        return x + self.fuse(h)


class MultiScaleResidualCNN(nn.Module):
    """Zero-initialized 1-channel residual vessel enhancer (exact receptive-field radius = 7 px)."""

    rf_radius = 7

    def __init__(self, channels: int = 24):
        super().__init__()
        self.stem = nn.Conv2d(1, channels, 3, padding=1)
        self.unit1 = DilatedResidualUnit(channels)
        self.unit2 = DilatedResidualUnit(channels)
        self.head = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.stem(x), inplace=True)
        h = self.unit1(h)
        h = self.unit2(h)
        return self.head(h)


def _inv_sigmoid(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


class EnhancementV3(nn.Module):
    supports_any_size = True  # tiles the small-RF residual CNN on huge images, so whole-image inference never OOMs
    accepts_fov = True        # uses the true FOV mask when supplied (and expects mask_fov=True in data config)

    def __init__(
        self,
        sigmas=(16.0, 48.0, 128.0),
        noise_floor: float = 0.05,
        weber_gain: float = 4.0,
        outlier_cap: float = 0.35,
        res_channels: int = 24,
        residual: bool = True,
        fusion: str = "tri_rg",
        raw_gate_init: float = 0.25,
    ):
        super().__init__()
        assert fusion in ("tri_rg", "gated_rgb", "mono"), f"unknown fusion mode {fusion!r}"
        self.fusion = fusion
        self.outlier_cap = float(outlier_cap)

        # Learnable RGB -> vessel-luminance mix, initialized to pure Green [0, 1, 0]
        self.mix = nn.Conv2d(3, 1, 1, bias=False)
        with torch.no_grad():
            self.mix.weight.copy_(torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1))

        self.illum = MultiScaleMaskedIllumination(sigmas=sigmas)

        # Noise floor bounded in (0.02, 0.15) to avoid dark-patch quantization noise blowup
        rel = (min(max(noise_floor, 0.021), 0.149) - 0.02) / 0.13
        self.floor_logit = nn.Parameter(torch.tensor(_inv_sigmoid(rel)))
        self.log_gain = nn.Parameter(torch.tensor(math.log(weber_gain)))
        if fusion != "mono":
            self.log_gain_red = nn.Parameter(torch.tensor(math.log(max(1.0, weber_gain * 0.75))))

        # Residual vessel refinement (can be set to None by edge_profile.py for pre- vs post-CNN inspection)
        self.residual = MultiScaleResidualCNN(res_channels) if residual else None

        # Gentle post-residual contrast curve around 0.5 (starts as exact identity)
        self.log_gamma = nn.Parameter(torch.zeros(1))
        self.log_alpha = nn.Parameter(torch.zeros(1))

        # Gated fusion parameters (start mostly enhanced while keeping ~25% raw intensity context)
        if fusion == "tri_rg":
            self.raw_gate_logit = nn.Parameter(torch.tensor(_inv_sigmoid(raw_gate_init)))
        elif fusion == "gated_rgb":
            self.rgb_gate_logits = nn.Parameter(torch.full((3,), _inv_sigmoid(raw_gate_init)))
        self.last = {}

    @staticmethod
    def fov_masks(x: torch.Tensor, fov: torch.Tensor | None = None):
        """Returns (raw_mask, stat_mask) where stat_mask erodes 2 px off the FOV rim for clean background stats."""
        if fov is not None:
            raw = (fov.to(dtype=x.dtype, device=x.device) > 0.5).float()
            if raw.ndim == 3:
                raw = raw.unsqueeze(1)
        else:
            # Fallback when called without explicit fov (assumes data.mask_fov=True zeroed outside-FOV pixels):
            # accept any non-zero pixel (> 0) and close tiny interior 0-holes so dark rims (HRF/FIVES) aren't clipped
            fg = (x.amax(dim=1, keepdim=True) > 1e-4).float()
            dil = F.max_pool2d(fg, kernel_size=9, stride=1, padding=4)
            raw = -F.max_pool2d(-dil, kernel_size=9, stride=1, padding=4)
        eroded = -F.max_pool2d(-raw, kernel_size=5, stride=1, padding=2)
        stat = torch.where(eroded.sum(dim=(-2, -1), keepdim=True) > 16.0, eroded, raw)
        return raw, stat

    def noise_floor(self) -> torch.Tensor:
        return 0.02 + 0.13 * torch.sigmoid(self.floor_logit)

    def _weber(self, ch: torch.Tensor, stat_mask: torch.Tensor, gain: torch.Tensor, floor: torch.Tensor):
        # Pass 1: coarse local background
        L0 = self.illum(ch, stat_mask).clamp(min=1e-4)
        # Soft-cap bright outliers (optic disc pallor, exudates, specular glints) so they cannot bleed into L
        excess = F.relu(ch - L0)
        cap = self.outlier_cap * L0 + 0.03
        ch_clipped = torch.minimum(ch, L0) + cap * torch.tanh(excess / cap)
        # Pass 2: outlier-free local illumination + noise-floored Weber contrast mapped to (0, 1)
        L = self.illum(ch_clipped, stat_mask).clamp(min=1e-4)
        c = (ch - L) / (L + floor)
        z = 0.5 + 0.5 * torch.tanh(gain * c)
        return z, L

    def _apply_residual(self, z: torch.Tensor, tiled: bool = False, tile: int = 1024) -> torch.Tensor:
        if self.residual is None:
            return z
        H, W = z.shape[-2:]
        if not tiled or H * W <= tile * tile:
            return (z + self.residual(z)).clamp(0.0, 1.0)
        # Exact halo tiling (halo >= rf_radius = 7 px gives bit-identical output to full-image convolution)
        halo = 16
        out = torch.empty_like(z)
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
                py0, px0 = max(0, y0 - halo), max(0, x0 - halo)
                py1, px1 = min(H, y1 + halo), min(W, x1 + halo)
                sub = z[..., py0:py1, px0:px1]
                res = (sub + self.residual(sub)).clamp(0.0, 1.0)
                out[..., y0:y1, x0:x1] = res[..., y0 - py0:y1 - py0, x0 - px0:x1 - px0]
        return out

    def _run(self, x: torch.Tensor, tiled_residual: bool = False, fov: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        raw_mask, stat_mask = self.fov_masks(x, fov)
        y = self.mix(x).clamp(0.0, 1.0)

        floor = self.noise_floor()
        gain_v = self.log_gain.exp().clamp(1.0, 12.0)
        z, L = self._weber(y, stat_mask, gain_v, floor)
        self.last = {"illumination": L.detach(), "pre_residual": z.detach()}

        z_enh = self._apply_residual(z, tiled=tiled_residual)

        # Gentle symmetric contrast curve around 0.5 -> [-1, 1] inside FOV
        gamma = self.log_gamma.exp().clamp(0.5, 2.0)
        alpha = self.log_alpha.exp().clamp(0.5, 2.5)
        centered = z_enh - 0.5
        enh_01 = (0.5 + alpha * torch.sign(centered) * (centered.abs() + 1e-6).pow(gamma)).clamp(0.0, 1.0)
        enh_v = (2.0 * enh_01 - 1.0) * raw_mask

        if self.fusion == "mono":
            return torch.cat([enh_v, enh_v, enh_v], dim=1)

        # Red-channel outlier-clipped Weber map preserves vessel/disc SNR when Green is attenuated in dark eyes
        gain_r = self.log_gain_red.exp().clamp(1.0, 12.0)
        z_red, _ = self._weber(x[:, 0:1], stat_mask, gain_r, floor)
        enh_r = (2.0 * z_red - 1.0) * raw_mask

        if self.fusion == "tri_rg":
            gate = torch.sigmoid(self.raw_gate_logit)
            raw_ctx = (2.0 * y - 1.0) * raw_mask
            ch2 = (1.0 - gate) * enh_v + gate * raw_ctx
            return torch.cat([enh_v, enh_r, ch2], dim=1)

        # 'gated_rgb': per-channel residual blend between enhanced stack and raw RGB
        gate3 = torch.sigmoid(self.rgb_gate_logits).view(1, 3, 1, 1)
        raw_rgb = (2.0 * x - 1.0) * raw_mask
        enh_stack = torch.cat([enh_v, enh_v, enh_r], dim=1)
        return (1.0 - gate3) * enh_stack + gate3 * raw_rgb

    def forward(self, x: torch.Tensor, fov: torch.Tensor | None = None) -> torch.Tensor:
        fp32 = (torch.autocast(device_type="cuda", enabled=False)
                if x.is_cuda and torch.is_autocast_enabled() else contextlib.nullcontext())
        with fp32:
            return self._run(x, tiled_residual=False, fov=fov)

    @torch.no_grad()
    def enhance_image(self, x: torch.Tensor, patch: int = 384, fov: torch.Tensor | None = None) -> torch.Tensor:
        """Whole-image inference path (1, 3, H, W): seamless normalization + exact halo-tiled residual CNN."""
        del patch  # purely spatial operators need no patch grid
        return self._run(x, tiled_residual=True, fov=fov)

    def describe(self) -> dict:
        w = self.mix.weight.detach().flatten().tolist()
        sw = self.illum.weights().detach().flatten().tolist()
        info = {
            "fusion": self.fusion,
            "colour_mix_rgb": [round(v, 3) for v in w],
            "scale_weights": {f"{int(s)}px": round(v, 3) for s, v in zip(self.illum.sigmas, sw)},
            "noise_floor": round(float(self.noise_floor()), 4),
            "weber_gain": round(float(self.log_gain.exp().clamp(1.0, 12.0)), 3),
            "gamma": round(float(self.log_gamma.exp().clamp(0.5, 2.0)), 3),
            "alpha": round(float(self.log_alpha.exp().clamp(0.5, 2.5)), 3),
        }
        if self.fusion == "tri_rg":
            info["weber_gain_red"] = round(float(self.log_gain_red.exp().clamp(1.0, 12.0)), 3)
            info["raw_gate"] = round(float(torch.sigmoid(self.raw_gate_logit)), 3)
        elif self.fusion == "gated_rgb":
            info["rgb_gates"] = [round(v, 3) for v in torch.sigmoid(self.rgb_gate_logits).detach().tolist()]
        return info
