# U²-Net-E `v2` — released checkpoints

This archive holds the 15 checkpoints used to produce every figure in the
[U2NetE_Research](https://github.com/tapos-datta/U2NetE_Research) repository: seed 1 of the three
main variants, on each of the four leave-one-dataset-out folds and on the `all` split used for the
external FIVES test.

```
runs/<variant>/<split>/<fold>/seed1/
    best.pt        weights, selected by validation Dice
    config.json    the exact configuration this run was trained with
    fold.json      the image lists of this fold
    env.json       package versions and hardware
```

| Variant | What it is |
| :--- | :--- |
| `rgb` | no enhancement (the reference) |
| `hybrid` | the original learned enhancement front-end |
| `hybrid_weber` | the final model: multi-scale Weber normalization |

| Split | Trained on | Intended test set |
| :--- | :--- | :--- |
| `lodo/lodo_CHASE_DB1` | the other three datasets | all 28 CHASE_DB1 images |
| `lodo/lodo_DRIVE` | the other three datasets | all 20 DRIVE images |
| `lodo/lodo_HRF` | the other three datasets | all 45 HRF images |
| `lodo/lodo_STARE` | the other three datasets | all 20 STARE images |
| `all/all` | all 113 pooled images | the 792 external FIVES images |

A checkpoint must be used with the matching fold. Running a LODO checkpoint on the dataset it was
trained on measures nothing.

## Use

```bash
git clone https://github.com/tapos-datta/U2NetE_Research
cd U2NetE_Research/v2
tar xzf u2nete-v2-weights-seed1.tar.gz          # creates runs/...
python evaluate.py runs/hybrid_weber/all/all/seed1 --external data/external_FIVES
```

Preparing the datasets first is described in `docs/DATA.md`. Verify the archive with:

```bash
sha256sum -c MANIFEST.sha256
```

## Three seeds, one published

Every headline result in the study is a three-seed mean, and the seed-to-seed spread matters: on
FIVES, `hybrid_weber` beats `rgb` in seeds 1 and 3 and loses in seed 2. Seed 1 alone will therefore
**not** reproduce the three-seed means. It reproduces the figures, which all use seed 1, and it is
enough to run the models. The remaining seeds can be requested, or retrained with
`run_experiments.py`.

No checkpoint is needed to check a published number: every run folder in the repository keeps its
`test/per_image.csv` and `test/summary.json`, and those are tracked in Git.

## Terms

These weights were trained on DRIVE, STARE, CHASE_DB1, HRF and FIVES, which are released for
research use. **The weights are therefore for research use, under the same restrictions as the
training data.** The Apache 2.0 licence of the source repository covers the code and documentation
and does **not** extend to these files. See `docs/DATA.md` for each dataset's terms.

Copyright 2026 Tapos Datta.
