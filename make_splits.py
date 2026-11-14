"""Create evaluation splits from data/pool/metadata.csv.

Protocols (each written to splits/<protocol>.json as a list of folds):
  lodo    Leave-one-dataset-out: train+val on 3 datasets, test on ALL images of the 4th.   -> cross-dataset generalization
  cv5     5-fold cross-validation, stratified by dataset (and HRF condition), grouped by patient.
          Every image is tested exactly once.                                               -> in-distribution accuracy
  legacy  The original run1 split (92/10/11), for continuity with the published numbers.
  all     Train on all 113 images (about 10% held back for validation), no internal test set.
          For the external test on FIVES (stage 3).                                          -> external generalization

Validation images always come from the training side of a fold (never from the test side) and are used
only for early stopping / checkpoint selection.

Usage:
    python make_splits.py --meta data/pool/metadata.csv --out splits --seed 0
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def load_meta(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def groups_by(rows, key="dataset"):
    """{dataset: {group: [ids]}}"""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r[key]][r["group"]].append(r["id"])
    return out


def pick_val(rows, frac, rng):
    """Choose validation groups from `rows`, stratified by dataset (at least 1 group per dataset)."""
    val = []
    for ds, groups in sorted(groups_by(rows).items()):
        names = sorted(groups)
        rng.shuffle(names)
        k = max(1, math.ceil(frac * len(names)))
        for g in names[:k]:
            val.extend(groups[g])
    return sorted(val)


def lodo(rows, val_frac, rng):
    folds = []
    for held_out in sorted({r["dataset"] for r in rows}):
        test = sorted(r["id"] for r in rows if r["dataset"] == held_out)
        rest = [r for r in rows if r["dataset"] != held_out]
        val = pick_val(rest, val_frac, rng)
        train = sorted(r["id"] for r in rest if r["id"] not in set(val))
        folds.append({"name": f"lodo_{held_out}", "held_out": held_out, "train": train, "val": val, "test": test})
    return folds


def cv(rows, k, val_frac, rng):
    # assign whole groups to folds, round-robin within each dataset/condition stratum
    fold_of_group = {}
    strata = defaultdict(list)
    for r in rows:
        strata[(r["dataset"], r["condition"])].append(r["group"])
    offset = 0
    for key in sorted(strata):
        groups = sorted(set(strata[key]))
        rng.shuffle(groups)
        for i, g in enumerate(groups):
            fold_of_group.setdefault(g, (i + offset) % k)
        offset += len(groups)  # rotate start so small strata don't all land in fold 0
    folds = []
    for f in range(k):
        test = sorted(r["id"] for r in rows if fold_of_group[r["group"]] == f)
        rest = [r for r in rows if fold_of_group[r["group"]] != f]
        val = pick_val(rest, val_frac, rng)
        train = sorted(r["id"] for r in rest if r["id"] not in set(val))
        folds.append({"name": f"cv{k}_fold{f}", "held_out": None, "train": train, "val": val, "test": test})
    return folds


def all_images(rows, val_frac, rng):
    val = pick_val(rows, val_frac, rng)
    train = sorted(r["id"] for r in rows if r["id"] not in set(val))
    return [{"name": "all", "held_out": None, "train": train, "val": val, "test": []}]


def legacy(rows, split_dir):
    ids = {r["id"] for r in rows}
    fold = {"name": "legacy_run1", "held_out": None}
    for part, fname in [("train", "run1_Train.txt"), ("val", "run1_Val.txt"), ("test", "run1_Test.txt")]:
        path = Path(split_dir) / fname
        if not path.exists():
            return None
        stems = [Path(line.strip()).stem for line in open(path) if line.strip()]
        missing = [s for s in stems if s not in ids]
        if missing:
            raise SystemExit(f"legacy split refers to images not in the pool: {missing[:5]}")
        fold[part] = sorted(stems)
    return [fold]


def check(folds, rows):
    group_of = {r["id"]: r["group"] for r in rows}
    for f in folds:
        parts = {p: {group_of[i] for i in f[p]} for p in ("train", "val", "test")}
        assert not parts["train"] & parts["test"], f"{f['name']}: patient in train and test"
        assert not parts["val"] & parts["test"], f"{f['name']}: patient in val and test"
        assert not parts["train"] & parts["val"], f"{f['name']}: patient in train and val"


def summarize(folds, rows):
    ds_of = {r["id"]: r["dataset"] for r in rows}
    for f in folds:
        def c(ids):
            d = defaultdict(int)
            for i in ids:
                d[ds_of[i]] += 1
            return " ".join(f"{k}:{v}" for k, v in sorted(d.items()))
        print(f"  {f['name']:18} train {len(f['train']):3} [{c(f['train'])}] | val {len(f['val']):2} | test {len(f['test']):3} [{c(f['test'])}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/pool/metadata.csv")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--seed", type=int, default=0, help="split seed (keep fixed; vary the training seed instead)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--legacy-dir", default="../data/splits")
    args = ap.parse_args()

    rows = load_meta(args.meta)
    rng = random.Random(args.seed)
    protocols = {"lodo": lodo(rows, args.val_frac, rng), f"cv{args.k}": cv(rows, args.k, args.val_frac, rng)}
    # own generator, so adding this protocol leaves the lodo/cv5 splits above exactly as they were
    protocols["all"] = all_images(rows, args.val_frac, random.Random(args.seed + 1))
    leg = legacy(rows, args.legacy_dir)
    if leg:
        protocols["legacy"] = leg

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for name, folds in protocols.items():
        check(folds, rows)
        with open(Path(args.out) / f"{name}.json", "w") as f:
            json.dump(folds, f, indent=1)
        print(f"{name}: {len(folds)} folds")
        summarize(folds, rows)


if __name__ == "__main__":
    main()
