import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepResidualUnit(nn.Module):
    """A standard bottleneck-free residual unit for 1-channel data."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 3x3 conv layers with padding=1 to maintain spatial resolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # 1x1 conv for skip connection if input and output channels differ
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return residual + x

class DeepResidualEnhance(nn.Module):
    """A 1-channel enhancement module using stacked residual units."""
    def __init__(self, channels=32):
        super().__init__()
        # Initial feature extraction
        self.input_conv = nn.Conv2d(1, channels, 3, padding=1)
        
        # Stacked Residual Units (Deepening the network)
        self.res_unit1 = DeepResidualUnit(channels, channels)
        self.res_unit2 = DeepResidualUnit(channels, channels)
        
        # Final output mapping back to 1 channel
        self.output_conv = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        # x is 1-channel Green image
        residual = x
        x = F.relu(self.input_conv(x))
        
        x = self.res_unit1(x)
        x = F.relu(x)
        
        x = self.res_unit2(x)
        x = F.relu(x)
        
        x = self.output_conv(x)
        # Learn the residual: Final Enhanced Image = Original Green + Learned Residual
        return residual + x



class ResidualEnhance(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 3, 1)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return residual + x  # residual enhancement

class LearnableContrast(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        # Per-channel parameters. Set to 1 for 1-channel Green.
        self.gamma = nn.Parameter(torch.ones(channels)) 
        self.alpha = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        
        # Multi-scale average pooling for local mean
        self.blur_small = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.blur_large = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        with torch.no_grad():
            self.blur_small.weight[:] = 1.0 / 9.0
            self.blur_large.weight[:] = 1.0 / 49.0

    def forward(self, x_enhanced):
        # x_enhanced is the 1-channel output from DeepResidualEnhance
        
        local_mean = 0.5 * self.blur_small(x_enhanced) + 0.5 * self.blur_large(x_enhanced)
        x_norm = x_enhanced - local_mean
        
        # Adaptive Gamma Correction
        gamma_view = self.gamma.view(1, -1, 1, 1)
        alpha_view = self.alpha.view(1, -1, 1, 1)
        beta_view = self.beta.view(1, -1, 1, 1)

        x_enh = torch.sign(x_norm) * torch.pow(torch.abs(x_norm) + 1e-6, gamma_view)
        
        # The enhancement is applied to the image centered around the local mean.
        # Output = (local_mean + enhanced_difference)
        return alpha_view * x_enh + local_mean + beta_view
    
class EnhancementModule(nn.Module):
    def __init__(self, mode: str = "hybrid", normalize=True, mean=0.5, std=0.5, use_minmax=True):
        super().__init__()
        assert mode in ["rgb", "green", "hybrid"], "mode must be one of ['rgb', 'green', 'hybrid']"
        self.mode = mode
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.use_minmax = use_minmax

        # Using the deeper, 1-channel optimized modules
        self.local_conv = DeepResidualEnhance(channels=32)
        self.contrast = LearnableContrast(channels=1)

        # Stage 6 ablation switches (v2 addition). The defaults reproduce the run1 module exactly, and no parameter
        # is added unless a switch needs one, so existing checkpoints load unchanged.
        self.norm = "minmax"      # minmax | masked | weber
        self.minmax_floor = 0.0   # added to the min-max denominator
        self.raw_rg = False       # feed [enhanced, raw green, raw red] instead of [enhanced] x 3
        self.cap_q = 0.995        # 'masked': quantile used as the stretch maximum

    def set_ablation(self, norm="minmax", floor=0.0, raw_rg=False, cap_q=0.995,
                     weber_sigmas=(16.0, 48.0, 128.0), weber_gain=4.0):
        """Switch on one stage 6 change (see PROTOCOL.md, stage 6)."""
        assert norm in ("minmax", "masked", "weber"), f"unknown norm {norm!r}"
        self.norm, self.minmax_floor, self.raw_rg, self.cap_q = norm, float(floor), bool(raw_rg), float(cap_q)
        if norm == "weber":
            from .EnhancementV3 import MultiScaleMaskedIllumination
            self.illum = MultiScaleMaskedIllumination(sigmas=tuple(float(v) for v in weber_sigmas))
            self.log_gain = nn.Parameter(torch.tensor(math.log(float(weber_gain))))
        return self

    def minmax_stretch(self, x, eps=1e-5):
        # Per-image min-max normalization for 1-channel data
        x_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + eps + self.minmax_floor)

    @staticmethod
    def _eye_mask(x):
        """Eye mask from an input whose outside-FOV pixels were set to 0 (data.mask_fov), as in EnhancementV3:
        any non-zero pixel, with small interior holes closed so very dark vessels are not treated as outside."""
        fg = (x.amax(dim=1, keepdim=True) > 1e-4).float()
        dil = F.max_pool2d(fg, kernel_size=9, stride=1, padding=4)
        return -F.max_pool2d(-dil, kernel_size=9, stride=1, padding=4) > 0.5

    def masked_stretch(self, g, x, eps=1e-5):
        """Min-max inside the eye only, with the maximum capped at the cap_q quantile (bright disc, exudates).
        Patches with almost no eye fall back to the plain min-max. Output clipped to [0, 1]."""
        B = g.shape[0]
        mask = self._eye_mask(x).reshape(B, -1)
        flat = g.float().reshape(B, -1)
        enough = mask.sum(dim=1) >= 64
        mask = torch.where(enough[:, None], mask, torch.ones_like(mask))
        lo = torch.where(mask, flat, torch.full_like(flat, float("inf"))).amin(dim=1)
        hi = torch.nanquantile(torch.where(mask, flat, torch.full_like(flat, float("nan"))), self.cap_q, dim=1)
        lo, hi = lo.view(B, 1, 1, 1), hi.view(B, 1, 1, 1)
        return ((g.float() - lo) / (hi - lo + eps)).clamp(0.0, 1.0)

    def weber(self, g):
        """Multi-scale Weber contrast over the patch (no eye mask, outlier cap or noise floor), mapped to (0, 1)."""
        g = g.float()
        L = self.illum(g, torch.ones_like(g)).clamp(min=1e-4)
        gain = self.log_gain.exp().clamp(1.0, 12.0)
        return 0.5 + 0.5 * torch.tanh(gain * (g - L) / (L + 1e-3))

    def forward(self, x):
        if self.mode == "rgb":
            return x
        
        # --- Start of Hybrid/Green Processing ---
        
        # Step 1: Extract 1-channel Green (most vessel contrast)
        # x[:, 1:2, :, :] extracts channel 1 (Green) and keeps it 4D (B, 1, H, W)
        g_channel = x[:, 1:2, :, :]
        
        if self.mode == "green":
            # Simple green channel extraction (triplicated for 3-channel backbone)
            return torch.cat([g_channel] * 3, dim=1)

        elif self.mode == "hybrid":
            # Step 2: Min-max stretch (optional); stage 6 variants replace or modify it
            if self.norm == "minmax":
                if self.use_minmax:
                    g_channel = self.minmax_stretch(g_channel)
            else:
                fp32 = (torch.autocast(device_type="cuda", enabled=False)
                        if x.is_cuda and torch.is_autocast_enabled() else contextlib.nullcontext())
                with fp32:
                    g_channel = self.masked_stretch(g_channel, x) if self.norm == "masked" else self.weber(g_channel)

            # Step 3: Local feature enhancement (Deeper CNN)
            # Input: 1-channel Green. Output: 1-channel enhanced image.
            x_local = self.local_conv(g_channel) 

            # Step 4: Learnable contrast (adaptive tone mapping)
            x_final_enhanced = self.contrast(x_local)

            # Step 5: Triplicate the enhanced 1-channel image for 3-channel backbone
            # (stage 6 'raw_rg': enhanced, raw green, raw red instead)
            if self.raw_rg:
                x_final_enhanced = torch.cat([x_final_enhanced, x[:, 1:2].to(x_final_enhanced.dtype),
                                              x[:, 0:1].to(x_final_enhanced.dtype)], dim=1)
            else:
                x_final_enhanced = torch.cat([x_final_enhanced] * 3, dim=1)

            # Step 6: Normalize for backbone
            if self.normalize:
                # Ensure mean/std match the 3-channel output
                x_final_enhanced = (x_final_enhanced - self.mean) / self.std

            return x_final_enhanced
