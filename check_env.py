"""Check a machine before running experiments: software, GPU, data, and real training speed.

Usage (on the server, inside the project's virtual environment):
    CUDA_VISIBLE_DEVICES=1 python check_env.py            # tests the GPU you picked
    python check_env.py --iters 30 --variant enh2c_scale_strong

It trains the real model for a few iterations on random data (no files are written) and estimates how long
each plan will take on this machine.
"""
import argparse
import importlib
import os
import platform
import sys
import time
from pathlib import Path

from u2nete.utils import ROOT, amp_dtype, autocast, configure_backend, get_device, load_config

import torch

from u2nete.losses import DeepSupervisionLoss
from u2nete.model import build_model

OK, BAD = "ok  ", "FAIL"


def line(status, msg):
    print(f"[{status}] {msg}")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="hybrid_scale_strong")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    problems = 0

    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
    for mod in ["numpy", "cv2", "skimage", "scipy", "PIL", "matplotlib"]:
        try:
            m = importlib.import_module(mod)
            line(OK, f"{mod} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            problems += 1
            line(BAD, f"{mod}: {e}  ->  pip install -r requirements.txt")

    # --- GPU
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set: all GPUs visible)')}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            line(OK, f"GPU {i}: {torch.cuda.get_device_name(i)}  {free / 2**30:.1f} of {total / 2**30:.1f} GB free")
    else:
        line("note", "no NVIDIA GPU visible to PyTorch" + (" (Apple GPU will be used)" if torch.backends.mps.is_available() else ""))
        if sys.platform.startswith("linux"):
            line("note", "on a GPU server this usually means the CPU-only torch build: reinstall torch for your CUDA version")
    device = get_device("auto")
    configure_backend(device)

    # --- data
    pool = ROOT / "data" / "pool"
    n = len(list((pool / "images").glob("*.png"))) if pool.exists() else 0
    line(OK if n == 113 else BAD, f"data/pool: {n} images (expected 113)")
    problems += n != 113
    for f in ["lodo.json", "cv5.json"]:
        exists = (ROOT / "splits" / f).exists()
        line(OK if exists else BAD, f"splits/{f}")
        problems += not exists

    # --- speed test with the real model and loss
    cfg = load_config(ROOT / "configs" / "variants" / f"{args.variant}.json")
    amp = amp_dtype(cfg["train"].get("amp", "auto"), device)
    model = build_model(cfg["model"]).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = DeepSupervisionLoss(cfg["loss"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    b, p = int(cfg["train"]["batch_size"]), int(cfg["data"]["patch_size"])
    x = torch.rand(b, 3, p, p, device=device)
    y = (torch.rand(b, 1, p, p, device=device) > 0.9).float()
    m = torch.ones_like(y)
    print(f"\nspeed test: {args.variant}, device {device}, mixed precision {amp if amp else 'off'}, batch {b} x {p}px")
    times = []
    for i in range(args.iters + 3):
        sync(device)
        t = time.time()
        with autocast(device, amp):
            out = model(x)
        loss = loss_fn([o.float() for o in out], y, m)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sync(device)
        if i >= 3:  # skip warm-up (cuDNN autotuning)
            times.append(time.time() - t)
    sec = sorted(times)[len(times) // 2]
    if device.type == "cuda":
        line(OK, f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    line(OK, f"{sec:.3f} s per iteration (GPU compute only; data loading adds a little)")

    iters = int(cfg["train"]["iterations"])
    per_run_h = sec * iters / 3600 * 1.15  # +15% for validation, data loading and the TTA test
    # stage 1: 16 ordinary runs + 8 that take ~2x (EnhancementV2 + consistency, full-resolution "hybrid")
    # stage 2 (36 new runs): half are EnhancementV2 + consistency
    print(f"\nupper-bound estimates (every run trains all {iters} iterations; early stopping usually ends sooner):")
    print(f"  one run of {args.variant}: ~{per_run_h:.1f} h")
    print(f"  stage1 (24 runs): ~{(16 + 8 * 2) * per_run_h:.0f} h")
    print(f"  stage2 (36 new runs): ~{(18 + 18 * 2) * per_run_h:.0f} h")
    print("\nall checks passed" if not problems else f"\n{problems} problem(s) above need fixing first")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
