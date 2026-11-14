"""Run a batch of experiments one after another; completed runs are skipped, so it is safe to stop and restart.

Usage:
    python run_experiments.py --plan stage1 --dry-run        # list runs and estimated time
    python run_experiments.py --plan stage1                  # run them
    python run_experiments.py --variants hybrid rgb --split lodo --seeds 1 2 3
    caffeinate -i python run_experiments.py --plan stage1    # keep the Mac awake during long batches

Plans (see PROTOCOL.md for the reasoning):
    smoke    1 short run per protocol to check the setup                          (~15 min)
    stage1   Does the enhancement help on an unseen dataset? LODO, 1 seed         (6 variants x 4 folds, ~2 days)
    stage1b  Controls for the winning recipe + scale x augmentation test (4 variants x 4 folds)
    stage2   Confirm hybrid vs rgb and CLAHE: 3 seeds LODO + 2 seeds 5-fold CV for hybrid/rgb (44 new runs)
    legacy   Reproduce the original 11-image test split with the new pipeline
    stage3   External test: train hybrid, rgb, clahe on all 113 images (3 seeds), then test each on FIVES
             (needs data/external_FIVES from prepare_data.py --layout fives; splits/all.json from make_splits.py)
    smoke_v3 Short EnhancementV3 check on the server: mixed precision, consistency loss, test step  (~15 min)
    stage4   Screen EnhancementV3 on unseen datasets: enh3 and its enh3_mono ablation, LODO, 1 seed (8 runs)
    stage4b  Confirm enh3: 3 seeds LODO and 2 seeds 5-fold CV, matching the stage 2 controls (18 new runs)
    stage5   External test of enh3: train on all 113 images (3 seeds), then test each on FIVES once
    smoke_s6 Short check of the 4 stage 6 ablation variants (1 fold, 60 iterations each)          (~10 min)
    stage6   Ablation: each EnhancementV3 change added to hybrid alone, LODO, 1 seed (16 runs)
    stage6_fives  Same 4 variants trained on all 113 images (seed 1), then tested on FIVES (exploratory)
    stage6b  Confirm hybrid_weber: LODO seeds 2 and 3 (8 new runs)
    stage6b_fives  hybrid_weber on all 113 images, seeds 2 and 3, each then tested on FIVES (2 new runs)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from u2nete.utils import load_config, load_json

STAGE6 = ["hybrid_masknorm", "hybrid_rg", "hybrid_floor", "hybrid_weber"]

PLANS = {
    "smoke": [("hybrid_scale_strong", "lodo", [0], [1], ["train.iterations=300", "train.val_every=150"]),
              ("hybrid_scale_strong", "cv5", [0], [1], ["train.iterations=300", "train.val_every=150"])],
    "stage1": [(v, "lodo", None, [1], []) for v in [
        "hybrid", "rgb_scale_strong", "clahe_scale_strong", "hybrid_scale_strong", "hybrid_scale_strong_fovnorm",
        "enh2c_scale_strong"]],
    # after stage 1: controls for the winning recipe (original resolution, basic augmentation) and a 2x2 test of
    # scale normalization x strong augmentation (with hybrid and hybrid_scale_strong from stage 1)
    "stage1b": [(v, "lodo", None, [1], []) for v in ["rgb", "clahe", "hybrid_strong", "hybrid_scale"]],
    # edit STAGE2_VARIANTS after stage 1: the best enhancement variant and its no-enhancement control
    "stage2": [(v, "lodo", None, [1, 2, 3], []) for v in ["hybrid", "rgb", "clahe"]]
              + [(v, "cv5", None, [1, 2], []) for v in ["hybrid", "rgb"]],
    "legacy": [(v, "legacy", None, [1, 2, 3], []) for v in ["v1_like", "hybrid_scale_strong_fovnorm"]],
    "stage3": [(v, "all", None, [1, 2, 3], []) for v in ["hybrid", "rgb", "clahe"]],
    "smoke_v3": [("enh3", "lodo", [1], [1], ["train.iterations=300", "train.val_every=150"]),
                 ("enh3c", "lodo", [1], [1], ["train.iterations=40", "train.val_every=20"])],
    # compared with the existing rgb and hybrid LODO runs (same folds, same seed), so no control is retrained here
    "stage4": [(v, "lodo", None, [1], []) for v in ["enh3", "enh3_mono"]],
    "stage4b": [("enh3", "lodo", None, [1, 2, 3], []), ("enh3", "cv5", None, [1, 2], [])],
    "stage5": [("enh3", "all", None, [1, 2, 3], [])],
    # stage 6: compared with the existing hybrid seed-1 runs on the same folds, so no control is retrained here
    "smoke_s6": [(v, "lodo", [1], [1], ["train.iterations=60", "train.val_every=30"]) for v in STAGE6],
    "stage6": [(v, "lodo", None, [1], []) for v in STAGE6],
    "stage6_fives": [(v, "all", None, [1], []) for v in STAGE6],
    # stage 6b: seed 1 exists from stage 6 and is skipped, so these train seeds 2 and 3 only
    "stage6b": [("hybrid_weber", "lodo", None, [1, 2, 3], [])],
    "stage6b_fives": [("hybrid_weber", "all", None, [1, 2, 3], [])],
}

# plans whose models are also tested on an external pool: (pool folder, output sub-folder in each run)
EXTERNAL = {"stage3": ("data/external_FIVES", "external_FIVES"),
            "stage5": ("data/external_FIVES", "external_FIVES"),
            "stage6_fives": ("data/external_FIVES", "external_FIVES"),
            "stage6b_fives": ("data/external_FIVES", "external_FIVES")}

# rough minutes per run on an M2 Pro (12k iterations max at 0.42 s/it with scale normalization, 0.87 s/it
# without; early stopping usually ends sooner; includes validation and the 8-fold TTA test)
# consistency-loss variants run a second enhancement view: measured 0.82 s/it
MINUTES = {"scale": 95, "native": 185, "consistency": 175, "external": 40}


def run_dir(variant, split, fold_name, seed, out):
    return Path(out) / variant / split / fold_name / f"seed{seed}"


def expand(plan_rows, out, external=None):
    jobs = []
    for variant, split, folds, seeds, extra in plan_rows:
        split_file = Path("splits") / f"{split}.json"
        all_folds = load_json(split_file)
        for f in (folds if folds is not None else range(len(all_folds))):
            for seed in seeds:
                d = run_dir(variant, split, all_folds[f]["name"], seed, out)
                # trained = its test summary exists (or, with no internal test set, train_summary.json);
                # done = trained and the external test (if the plan has one) is finished too
                trained = ((d / "test" / "summary.json").exists() if all_folds[f]["test"]
                           else (d / "train_summary.json").exists())
                ext_done = external is None or (d / external[1] / "summary.json").exists()
                jobs.append({"variant": variant, "split": str(split_file), "fold": f, "seed": seed,
                             "extra": extra, "dir": d, "trained": trained, "done": trained and ext_done})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(PLANS))
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--split", default="lodo")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides passed to every run, e.g. train.workers=2")
    args = ap.parse_args()

    rows = PLANS[args.plan] if args.plan else [(v, args.split, None, args.seeds, []) for v in args.variants or []]
    if not rows:
        ap.error("give --plan or --variants")
    out = "runs_smoke" if args.plan in ("smoke", "smoke_v3", "smoke_s6") else args.out
    external = EXTERNAL.get(args.plan)
    if external and not (Path(external[0]) / "metadata.csv").exists():
        ap.error(f"{external[0]}/metadata.csv not found: run prepare_data.py --layout fives first (README, stage 3)")
    jobs = expand(rows, out, external)
    todo = [j for j in jobs if not j["done"]]

    est = 0
    for j in todo:
        cfg = load_config(Path("configs/variants") / f"{j['variant']}.json")
        if external:
            est += MINUTES["external"]
        if j["trained"]:
            continue
        if j["extra"]:
            est += 5
        elif cfg["loss"].get("consistency_weight", 0) > 0:
            est += MINUTES["consistency"]
        else:
            est += MINUTES["scale" if cfg["data"].get("scale_norm") else "native"]
    print(f"{len(jobs)} runs in plan, {len(jobs) - len(todo)} already done, {len(todo)} to run "
          f"(~{est / 60:.1f} h on an M2 Pro, usually much less on an NVIDIA GPU; check_env.py measures yours)")
    for j in jobs:
        print(f"  [{'done' if j['done'] else 'todo'}] {j['dir']}")
    if args.dry_run:
        return

    for n, j in enumerate(todo, 1):
        if j["trained"]:
            print(f"\n=== [{n}/{len(todo)}] {j['dir']} (already trained; external test only) ===", flush=True)
        else:
            cmd = [sys.executable, "train.py", "--config", f"configs/variants/{j['variant']}.json",
                   "--split", j["split"], "--fold", str(j["fold"]), "--seed", str(j["seed"]), "--out", out]
            if j["extra"] or args.set:
                cmd += ["--set", *j["extra"], *args.set]
            resume = (j["dir"] / "last.pt").exists()
            if resume:
                cmd.append("--resume")
            print(f"\n=== [{n}/{len(todo)}] {j['dir']}{' (resuming)' if resume else ''} ===", flush=True)
            run_step(cmd, "training")
        if external:
            run_step([sys.executable, "evaluate.py", str(j["dir"]), "--external", external[0],
                      "--name", external[1]], f"external test ({external[1]})")


def run_step(cmd, what):
    t = time.time()
    code = subprocess.call(cmd)
    print(f"=== {what} finished in {(time.time() - t) / 60:.1f} min (exit {code}) ===", flush=True)
    if code != 0:
        print("stopping: fix the error above, then rerun the same command to continue")
        sys.exit(code)


if __name__ == "__main__":
    main()
