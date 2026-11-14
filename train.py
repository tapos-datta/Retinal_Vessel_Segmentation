"""Train one model on one fold, select the checkpoint by validation Dice, then evaluate on the fold's test images.

Usage:
    python train.py --config configs/variants/hybrid.json --split splits/lodo.json --fold 0 --seed 1
    python train.py --config configs/variants/hybrid.json --split splits/lodo.json --fold 0 --seed 1 \
        --set train.iterations=500 train.val_every=250        # quick check

Output: runs/<variant>/<split>/<fold>/seed<seed>/
    config.json  fold.json  env.json  train_log.csv  val_log.csv  best.pt  last.pt  test/per_image.csv  test/summary.json
"""
import argparse
import csv
import math
import time
from pathlib import Path

from u2nete.utils import (ROOT, amp_dtype, autocast, configure_backend, environment_info, get_device, load_config,
                          load_json, parse_overrides, save_json, seed_everything)

import numpy as np
import torch
from torch.utils.data import DataLoader

from u2nete.data import PatchDataset, load_metadata, load_samples
from u2nete.infer import predict_sample
from u2nete.losses import DeepSupervisionLoss
from u2nete.metrics import compute
from u2nete.model import build_model, count_parameters


def lr_at(it, cfg):
    t = cfg["train"]
    if it < t["warmup"]:
        return t["lr"] * (it + 1) / t["warmup"]
    progress = (it - t["warmup"]) / max(1, t["iterations"] - t["warmup"])
    return t["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


@torch.no_grad()
def validate(model, samples, cfg, device):
    """Mean FOV Dice over validation images at working scale (no TTA, for speed)."""
    model.eval()
    scores = {}
    for s in samples:
        prob = predict_sample(model, s, cfg, device, tta=False, to_original=False)
        scores[s.id] = compute(prob, s.gt, s.fov, cfg["inference"]["threshold"], with_cldice=False)["dice"]
    model.train()
    return float(np.mean(list(scores.values()))), scores


def run_dir_for(config_path, split_path, fold_name, seed, out_root):
    variant = Path(config_path).stem
    return Path(out_root) / variant / Path(split_path).stem / fold_name / f"seed{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--set", nargs="*", default=[], help="override config values, e.g. train.iterations=500")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    folds = load_json(args.split)
    fold = folds[args.fold]
    out = run_dir_for(args.config, args.split, fold["name"], args.seed, args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg["train"]["device"])
    configure_backend(device, cfg["train"].get("gpu_memory_fraction"))
    amp = amp_dtype(cfg["train"].get("amp", "auto"), device)
    seed_everything(args.seed)
    save_json(cfg, out / "config.json")
    save_json(fold, out / "fold.json")
    save_json({**environment_info(device), "seed": args.seed, "split": args.split, "fold": args.fold,
               "config_file": args.config}, out / "env.json")

    pool = (ROOT / cfg["data"]["pool"]) if not Path(cfg["data"]["pool"]).is_absolute() else Path(cfg["data"]["pool"])
    meta = load_metadata(pool)
    t0 = time.time()
    train_samples = load_samples(fold["train"], pool, cfg["data"], meta)
    val_samples = load_samples(fold["val"], pool, cfg["data"], meta)
    print(f"[{out}] device={device} train={len(train_samples)} val={len(val_samples)} "
          f"test={len(fold['test'])} (loaded in {time.time() - t0:.1f}s)")

    model = build_model(cfg["model"]).to(device)
    print(f"model: {cfg['model']['enhancement']} enhancement, {count_parameters(model) / 1e6:.2f}M parameters")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = DeepSupervisionLoss(cfg["loss"])
    # float16 needs loss scaling to avoid underflow; bfloat16 and full precision don't
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    print(f"mixed precision: {amp if amp else 'off'}")

    start_it, best, bad_evals = 0, -1.0, 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_it, best, bad_evals = ck["iteration"], ck["best"], ck["bad_evals"]
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        print(f"resumed at iteration {start_it} (best val Dice {best:.4f})")

    ds = PatchDataset(train_samples, cfg, seed=args.seed * 1000 + start_it)
    workers = int(cfg["train"]["workers"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], num_workers=workers,
                        worker_init_fn=ds.worker_init if workers else None,
                        prefetch_factor=4 if workers else None, pin_memory=device.type == "cuda")
    batches = iter(loader)  # one endless "epoch"; shut down explicitly at the end

    train_log = open(out / "train_log.csv", "a", newline="")
    val_log = open(out / "val_log.csv", "a", newline="")
    tw, vw = csv.writer(train_log), csv.writer(val_log)
    if start_it == 0:
        tw.writerow(["iteration", "loss", "consistency", "lr", "seconds"])
        vw.writerow(["iteration", "val_dice", "best", *[s.id for s in val_samples]])

    model.train()
    it, running, running_cons, t_last = start_it, 0.0, 0.0, time.time()
    cons_w = float(cfg["loss"].get("consistency_weight", 0.0))
    total = int(cfg["train"]["iterations"])
    while it < total:
        batch = next(batches)
        imgs, gts, fovs = batch[:3]
        for g in opt.param_groups:
            g["lr"] = lr_at(it, cfg)
        nb = device.type == "cuda"
        imgs, gts, fovs = imgs.to(device, non_blocking=nb), gts.to(device, non_blocking=nb), fovs.to(device, non_blocking=nb)
        # the network runs in mixed precision (if enabled); losses are computed in float32 for stability
        accepts_fov = getattr(model.enhance, "accepts_fov", False)
        with autocast(device, amp):
            if cons_w > 0:
                # enhancement consistency: two colour/illumination versions of the same patch should enhance alike
                img_b = batch[3].to(device, non_blocking=nb)
                enh_a = model.enhance(imgs, fov=fovs) if accepts_fov else model.enhance(imgs)
                outputs = model.u2net((enh_a - model.mean) / model.std if model.normalize else enh_a)
                enh_b = model.enhance(img_b, fov=fovs) if accepts_fov else model.enhance(img_b)
            elif accepts_fov:
                enh_a = model.enhance(imgs, fov=fovs)
                outputs = model.u2net((enh_a - model.mean) / model.std if model.normalize else enh_a)
            else:
                outputs = model(imgs)
        outputs = [o.float() for o in outputs]
        if cons_w > 0:
            enh_a, enh_b = enh_a.float(), enh_b.float()
            cons = (((enh_a - enh_b) ** 2) * fovs).sum() / (fovs.sum() * enh_a.shape[1]).clamp(min=1)
            loss = loss_fn(outputs, gts, fovs) + cons_w * cons
            running_cons += cons.item()
        else:
            loss = loss_fn(outputs, gts, fovs)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item()
        it += 1

        if it % cfg["train"]["log_every"] == 0:
            dt = time.time() - t_last
            avg = running / cfg["train"]["log_every"]
            cavg = running_cons / cfg["train"]["log_every"]
            tw.writerow([it, f"{avg:.5f}", f"{cavg:.5f}", f"{opt.param_groups[0]['lr']:.2e}", f"{dt:.1f}"])
            train_log.flush()
            print(f"it {it:6d}/{total}  loss {avg:.4f}" + (f"  consistency {cavg:.4f}" if cons_w > 0 else "")
                  + f"  lr {opt.param_groups[0]['lr']:.1e}  "
                  f"{dt / cfg['train']['log_every']:.3f}s/it")
            running, running_cons, t_last = 0.0, 0.0, time.time()

        if it % cfg["train"]["val_every"] == 0 or it == total:
            vd, per = validate(model, val_samples, cfg, device)
            improved = vd > best + 1e-4
            if improved:
                best, bad_evals = vd, 0
                torch.save({"model": model.state_dict(), "iteration": it, "val_dice": vd, "config": cfg}, out / "best.pt")
            else:
                bad_evals += 1
            vw.writerow([it, f"{vd:.4f}", f"{best:.4f}", *[f"{per[s.id]:.4f}" for s in val_samples]])
            val_log.flush()
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                        "iteration": it, "best": best, "bad_evals": bad_evals}, out / "last.pt")
            print(f"  validation Dice {vd:.4f}  (best {best:.4f}{', saved' if improved else ''})")
            t_last = time.time()
            if bad_evals >= cfg["train"]["patience"]:
                print(f"early stop: no improvement in {bad_evals} validations")
                break

    if hasattr(batches, "_shutdown_workers"):
        batches._shutdown_workers()  # otherwise the worker processes keep Python alive on macOS
    del batches, loader
    train_log.close()
    val_log.close()
    save_json({"stopped_at": it, "best_val_dice": best, "minutes": round((time.time() - t0) / 60, 1)},
              out / "train_summary.json")

    if not args.no_test and fold["test"]:  # the 'all' split has no internal test set
        from evaluate import evaluate_run
        evaluate_run(out, tta=cfg["inference"].get("tta_test", True))


if __name__ == "__main__":
    import os
    import sys
    main()
    # Python 3.13 on macOS: the multiprocessing resource tracker started by DataLoader workers can block
    # interpreter shutdown indefinitely. Everything is saved and closed at this point, so exit directly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
