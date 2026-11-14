import contextlib
import json
import os
import platform
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def get_device(name: str = "auto") -> torch.device:
    """NVIDIA GPU (cuda), then Apple GPU (mps), then CPU. Choose the NVIDIA card with CUDA_VISIBLE_DEVICES."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_backend(device: torch.device, memory_fraction=None) -> None:
    """Speed settings for NVIDIA GPUs (fixed 384 px patches suit cuDNN autotuning).

    memory_fraction caps this process's share of GPU memory, so a shared GPU keeps room for other jobs.
    """
    if device.type == "cuda":
        if memory_fraction:
            # needs an explicit index: torch.device("cuda") has none and recent PyTorch rejects it
            index = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(float(memory_fraction), index)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def amp_dtype(setting, device: torch.device):
    """Mixed precision: "auto" uses bfloat16 (or float16) on NVIDIA GPUs and full precision elsewhere."""
    if setting in (False, "off", "false", None) or device.type != "cuda":
        return None
    if setting in ("bf16", "bfloat16") or (setting in ("auto", True) and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def autocast(device: torch.device, dtype):
    """Mixed-precision context on NVIDIA GPUs; a no-op otherwise (avoids warnings on Apple GPUs and CPUs)."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def deep_update(base: dict, update: dict) -> dict:
    out = deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(pairs) -> dict:
    """Turn ["train.lr=1e-3", "data.scale_norm=true"] into a nested dict."""
    out: dict = {}
    for pair in pairs or []:
        key, raw = pair.split("=", 1)
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def load_config(path, overrides=None) -> dict:
    """Load a config; a config may name a parent with "extends" (path relative to itself)."""
    path = Path(path)
    cfg = load_json(path)
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_update(load_config(path.parent / parent), cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def environment_info(device: torch.device) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "git_commit": commit,
    }
