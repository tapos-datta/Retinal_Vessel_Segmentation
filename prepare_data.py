"""Pool all labelled images into one folder with FOV masks and a metadata table.

Usage:
    python prepare_data.py --src ../data/dataset --out data/pool
    python prepare_data.py --src ../data/dataset --out data/pool --official-fov /path/to/DRIVE/training/mask /path/to/HRF/mask

External test set FIVES (never used for training; see PROTOCOL.md, stage 3):
    python prepare_data.py --layout fives --src "../external_data/FIVES/extracted/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation" --out data/external_FIVES
    add --limit 2 for a quick check (2 images per condition and split)

--src may be a folder with Train/Val/Test sub-folders (each with images/ and masks/) or a single images/+masks/ folder.
Images and masks are paired by sorted order within each folder, as in the original pipeline.

--layout fives reads FIVES's own layout (train|test / Original|Ground truth, masks with the same file name).
Ids get the split as prefix (FIVES_train_10_A), because 50 file names occur in both splits as different images,
and exact duplicate images (6 pairs, found by pixel hash) are kept only once.
"""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from u2nete.io import IMAGE_EXTS, auto_fov, fov_diameter, identify, read_binary, read_rgb, write_png


def find_pairs(src: Path):
    folders = [p for p in sorted(src.iterdir()) if (p / "images").is_dir() and (p / "masks").is_dir()]
    if (src / "images").is_dir():
        folders = [src]
    for folder in folders:
        if folder.name.lower().startswith("val_aug"):
            continue  # augmented copies are regenerated on the fly; never pool them
        imgs = sorted(p for p in (folder / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
        masks = sorted(p for p in (folder / "masks").iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if len(imgs) != len(masks):
            raise SystemExit(f"{folder}: {len(imgs)} images but {len(masks)} masks")
        yield from zip(imgs, masks)


FIVES_CONDITIONS = {"A": "amd", "D": "diabetic_retinopathy", "G": "glaucoma", "N": "normal"}


def find_pairs_fives(src: Path, limit=None):
    """Yield (image, mask, meta) for the FIVES layout."""
    for split in ("train", "test"):
        img_dir, gt_dir = src / split / "Original", src / split / "Ground truth"
        if not img_dir.is_dir():
            raise SystemExit(f"{img_dir} not found: --src must be the folder that contains train/ and test/")
        per_cond = {}
        for img in sorted(img_dir.iterdir(), key=lambda p: (len(p.stem), p.stem)):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            cond = img.stem.split("_")[-1]
            per_cond[cond] = per_cond.get(cond, 0) + 1
            if limit and per_cond[cond] > limit:
                continue
            mask = gt_dir / img.name
            if not mask.exists():
                raise SystemExit(f"no ground truth for {img}")
            iid = f"FIVES_{split}_{img.stem}"
            yield img, mask, {"id": iid, "dataset": "FIVES", "group": iid,
                              "condition": FIVES_CONDITIONS.get(cond, cond)}


def find_official_fov(stem: str, dirs):
    for d in dirs:
        for p in Path(d).iterdir():
            # DRIVE: 21_training_mask.gif, HRF: 01_h_mask.tif
            if p.stem in (stem + "_mask", stem):
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", default="data/pool")
    ap.add_argument("--official-fov", nargs="*", default=[], help="folders with official FOV masks (DRIVE, HRF)")
    ap.add_argument("--layout", choices=["pool", "fives"], default="pool",
                    help="pool: images/ + masks/ folders (the 4 training datasets); fives: FIVES's own layout")
    ap.add_argument("--limit", type=int, default=None, help="fives only: images per condition and split (quick test)")
    args = ap.parse_args()

    out = Path(args.out)
    rows, seen, hashes, dropped, blank = [], set(), {}, [], []
    for src in args.src:
        if args.layout == "fives":
            pairs = find_pairs_fives(Path(src), args.limit)
        else:
            pairs = ((i, m, identify(i.name)) for i, m in find_pairs(Path(src)))
        for img_path, mask_path, meta in pairs:
            if meta["id"] in seen:
                raise SystemExit(f"duplicate image id {meta['id']}")
            seen.add(meta["id"])

            rgb = read_rgb(img_path)
            digest = hashlib.md5(rgb.tobytes()).hexdigest()
            if digest in hashes:  # the same picture under another name: keep the first copy only
                dropped.append(f"{meta['id']} (= {hashes[digest]})")
                print(f"skip     {meta['id']:22} identical to {hashes[digest]}")
                continue
            hashes[digest] = meta["id"]
            gt = read_binary(mask_path)
            if args.layout == "fives" and gt.sum() == 0:  # 2 FIVES images have an all-black label: unusable as a test
                blank.append(meta["id"])
                print(f"skip     {meta['id']:22} blank label")
                continue
            if gt.shape != rgb.shape[:2]:
                raise SystemExit(f"{img_path.name}: mask shape {gt.shape} != image {rgb.shape[:2]}")

            official = find_official_fov(meta["id"], args.official_fov) if args.official_fov else None
            fov = read_binary(official) if official else auto_fov(rgb)
            fov_src = "official" if official else "auto"

            write_png(out / "images" / f"{meta['id']}.png", rgb)
            write_png(out / "masks" / f"{meta['id']}.png", gt * 255)
            write_png(out / "fov" / f"{meta['id']}.png", fov * 255)

            rows.append({
                **meta,
                "source_file": f"{img_path.parent.parent.name}/{img_path.name}" if args.layout == "fives" else img_path.name,
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "fov_diameter": fov_diameter(fov),
                "fov_source": fov_src,
                "vessel_fraction_in_fov": round(float(gt[fov > 0].mean()), 4),
                "vessel_pixels": int(gt.sum()),
                "gt_outside_fov": int((gt * (1 - fov)).sum()),
            })
            print(f"{meta['dataset']:9} {meta['id']:22} {rgb.shape[1]}x{rgb.shape[0]}  FOV {rows[-1]['fov_diameter']}px ({fov_src})")

    rows.sort(key=lambda r: (r["dataset"], r["id"]))
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} images written to {out}")
    if dropped:
        print(f"{len(dropped)} exact duplicates skipped: {', '.join(dropped)}")
    if blank:
        print(f"{len(blank)} images with a blank label skipped: {', '.join(blank)}")
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        print(f"  {ds:9} n={len(sub):3}  mean FOV diameter {np.mean([r['fov_diameter'] for r in sub]):.0f}px")
    leak = [r["id"] for r in rows if r["gt_outside_fov"] > 0.01 * r["vessel_pixels"]]
    if leak:
        print(f"\nWARNING: >1% of vessel labels fall outside the FOV in {leak}; check those masks in {out/'fov'}")


if __name__ == "__main__":
    main()
