"""Model construction. The architecture code in models/ is the unchanged run1 code."""
import torch

from .models.EnhancementV2 import EnhancementV2
from .models.EnhancementV3 import EnhancementV3
from .models.U2NetArc import U2NET_lite
from .models.U2NetE import U2NetE


def build_model(model_cfg: dict) -> torch.nn.Module:
    enh = model_cfg["enhancement"]
    custom = enh in ("v2", "v3")
    model = U2NetE(
        base_model=U2NET_lite(),
        enhancement_mode="rgb" if custom else enh,            # rgb | green | hybrid | v2 | v3
        normalize=bool(model_cfg.get("imagenet_norm", False)),
    )
    if enh == "v2":
        opts = model_cfg.get("enh2", {})
        model.enhance = EnhancementV2(illum_sigma=float(opts.get("illum_sigma", 48)),
                                      curve_iters=int(opts.get("curve_iters", 4)),
                                      residual=bool(opts.get("residual", True)))
        return model
    if enh == "v3":
        opts = model_cfg.get("enh3", {})
        model.enhance = EnhancementV3(
            sigmas=tuple(opts.get("sigmas", (16.0, 48.0, 128.0))),
            noise_floor=float(opts.get("noise_floor", 0.05)),
            weber_gain=float(opts.get("weber_gain", 4.0)),
            outlier_cap=float(opts.get("outlier_cap", 0.35)),
            res_channels=int(opts.get("res_channels", 24)),
            residual=bool(opts.get("residual", True)),
            fusion=str(opts.get("fusion", "tri_rg")),
            raw_gate_init=float(opts.get("raw_gate_init", 0.25)),
        )
        return model
    # run1 behaviour: hybrid mode min-max stretches every 384 px patch independently
    model.enhance.use_minmax = bool(model_cfg.get("patch_minmax", True))
    if model_cfg.get("ablation"):  # stage 6: one EnhancementV3 change added to the hybrid module
        model.enhance.set_ablation(**model_cfg["ablation"])
    return model


def prediction(outputs, which: str) -> torch.Tensor:
    """U2NET returns [fused, side1, ..., side6] (all sigmoid).

    The original test.py used outputs[1] (side1). "fused" is the standard U²-Net prediction.
    """
    return outputs[0] if which == "fused" else outputs[1]


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
