# Datasets: Provenance, Terms and Preparation

No fundus image, label or FOV mask is distributed in this repository. Only
`data/pool/metadata.csv`, the split files and the result tables are tracked. This document lists
where each dataset comes from, what it may be used for, and exactly how it was prepared here.

---

## 1. Sources and terms

Each dataset is released by its own authors under its own terms. **Obtain each one from its
official source and comply with its licence**; the Apache 2.0 licence of this repository covers
the code only. Terms change, so check the current page rather than relying on this table.

| Dataset | Images used | Source | Access and terms |
| :--- | :-: | :--- | :--- |
| **DRIVE** | 20 (training split) | Digital Retinal Images for Vessel Extraction, Staal et al., 2004 | registration required; research use |
| **STARE** | 20 | STructured Analysis of the Retina, Hoover et al., 2000 | public download; research use |
| **CHASE_DB1** | 28 | Child Heart and Health Study in England, Fraz et al., 2012 | public download; research use |
| **HRF** | 45 | High-Resolution Fundus database, Budai et al., 2013 | public download; research use with citation |
| **FIVES** (external) | 792 of 800 | Fundus Image Vessel Segmentation dataset, Jin et al., *Scientific Data*, 2022 | public repository; CC licence stated by the authors |

Two further datasets would have provided a second external test and could not be obtained during
the study: **IOSTAR** (the download page reported a fault and asks users to e-mail the authors) and
**DR HAGIS** (the page did not load).

---

## 2. How the training pool was built

`prepare_data.py --src <core datasets> --out data/pool` produces 113 images in one folder with a
FOV mask and a metadata row each.

| Dataset | Images | Size (px) | Mean FOV diameter (px) | Vessel pixels in FOV | Labels used | Patient grouping |
| :--- | :-: | :-: | :-: | :-: | :--- | :--- |
| HRF | 45 (15 healthy, 15 DR, 15 glaucoma) | 3504 × 2336 | 3376 | 9.1% | official | one image per subject |
| CHASE_DB1 | 28 (14 children, both eyes) | 999 × 960 | 918 | 10.1% | first observer (`1stHO`) | both eyes kept together |
| DRIVE | 20 (training split only) | 565 × 584 | 542 | 12.6% | `manual1` | one image per subject |
| STARE | 20 | 700 × 605 | 666 | 10.3% | Hoover (`.ah`) | one image per subject |

Two properties of the pool shape everything else:

- **Eye size differs six-fold**, from 542 px in DRIVE to 3376 px in HRF, so a vessel of a given
  physical width spans six times as many pixels in HRF, and a 384 px patch covers 70% of a DRIVE eye
  but only 12% of an HRF eye.
- **HRF dominates by count** (45 of 113, each very large), so sampling is balanced per dataset
  during training.

### FOV masks

Estimated from the red channel: largest connected component, holes filled, no erosion. Checked
visually; under 1% of vessel labels fall outside them. Pass `--official-fov <dir> <dir>` to use the
official DRIVE and HRF masks instead.

> **Known property of the HRF masks.** In all 45 HRF images the estimated mask reaches the top,
> bottom and right edges of the picture, and on the right it extends past the visible eye into the
> dark border, so a little non-retina area is scored. This affects every variant equally, but it
> should be remembered whenever an HRF number is quoted.

### Verifying a rebuild

`data/pool/MANIFEST.sha256` lists the SHA-256 of all 339 files the preparation step writes (113
images, labels and FOV masks). After rebuilding, check it:

```bash
cd data/pool && sha256sum -c MANIFEST.sha256        # macOS: shasum -a 256 -c MANIFEST.sha256
```

Every line must report `OK`. A mismatch means the source files, the converter or the FOV estimation
differ from the ones used here, and the numbers in [RESULTS.md](RESULTS.md) will not reproduce
exactly. The manifest contains hashes only; no image is distributed.

### Second observer

Second-observer labels exist for STARE and CHASE_DB1 but were **not** added to the pool. Comparing
models against human-to-human agreement remains open work.

---

## 3. How FIVES was prepared

`prepare_data.py --layout fives --src <FIVES> --out data/external_FIVES` gives 792 scorable images
from 800:

- Identifiers carry FIVES's own split (for example `FIVES_test_10_A`). Fifty names occur in both of
  its splits as **different** images; both splits are used as test images, because none are used for
  training.
- **Six exact duplicate pairs** (identical pixels, found by hash) are kept once.
- **Two images with an all-black label** (`train/447_G`, `train/448_G`) are left out: with no vessel
  pixels, Dice and AUC are undefined. Every aggregate in this repository excludes them.
- FOV masks are estimated as for the training pool. In seven images 1.1–1.8% of the vessel labels
  fall just outside the estimated FOV, where the annotation was traced to the rim; those pixels are
  excluded like any pixel outside the FOV.

`data/external_FIVES/MANIFEST.sha256` lists the SHA-256 of the 2,382 files of the prepared external
set and is checked the same way.

FIVES was never used for training, validation or checkpoint selection. Its **first** use, in stage
3, was a pure external test. After stage 3 its results informed the design of EnhancementV3 and the
stage 6 ablation, so every later FIVES number is exploratory in that sense and is labelled as such.

---

## 4. Labelling conventions differ, and it shows in the metrics

The datasets were labelled by different people following different, rarely documented instructions.
Relative to eye size, FIVES draws vessels **2.6 times as wide as HRF** (6.08 against 2.30, in units
of width divided by FOV diameter × 1000). In pixels the ordering is different again, because pixel
width also follows image resolution.

The consequence is measurable. Predicted width divided by labelled width, averaged over three seeds,
runs from **+22% on CHASE_DB1 to −32% on FIVES**, for every variant, learned or not. A cross-dataset
Dice therefore contains a term that has nothing to do with whether a vessel was found, and that term
changes sign between datasets. Width-tolerant metrics (centreline recall and precision, clDice)
should be reported alongside Dice in any cross-dataset comparison; see
[RESULTS.md](RESULTS.md) §3B.

---

## 5. Obtaining the prepared data and weights

Raw images and `.pt` checkpoints are excluded from Git. Two routes:

1. **Rebuild from the public sources.** Download the five datasets from their official
   distributions and run `prepare_data.py` and `make_splits.py` as in
   [REPRODUCIBILITY.md](REPRODUCIBILITY.md). `data/pool/metadata.csv` and `splits/` are tracked, so
   a rebuild can be checked against them image by image.
2. **Ask for help if the rebuild does not match.** Open an issue. The images themselves cannot be
   shared: several of the datasets do not permit redistribution, and that applies to the converted
   copies in `data/pool` as much as to the originals. What can be offered is help with the
   preparation step and the manifests above to find where a rebuild diverged.

### Trained weights

Weights are not in Git either; `runs/` is about 2.7 GB. A curated subset, the 15 checkpoints behind
every figure, is attached to the
[latest release](https://github.com/tapos-datta/Retinal_Vessel_Segmentation/releases) as
`retinal-vessel-seg-v2-weights-seed1.tar.gz` (66 MB) and can be downloaded without any request or
approval; see [WEIGHTS.md](WEIGHTS.md). They carry their own terms: the models were trained on
research-use datasets, so **the weights are for research use under the same restrictions as the
training data**, and the Apache 2.0 licence of this repository does not extend to them.

Note that the weights are not needed to check any published number. Every run folder keeps its
`config.json`, `fold.json`, `env.json` and `test/per_image.csv`, and those are tracked, so each
result can be recomputed from the stored per-image metrics without downloading a single
checkpoint.
