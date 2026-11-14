# Stage 6b: hybrid_weber over 3 seeds

## LODO (8-fold TTA)

| Variant | CHASE_DB1 | DRIVE | HRF | STARE | Mean | Seed 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| `hybrid_weber` | 0.7843 | 0.7901 | 0.7915 | 0.7918 | 0.7894 | 0.7817 / 0.7942 / 0.7924 |
| `hybrid` | 0.8011 | 0.7925 | 0.8084 | 0.7854 | 0.7969 | 0.7983 / 0.7955 / 0.7968 |
| `rgb` | 0.7857 | 0.8057 | 0.7801 | 0.7689 | 0.7851 | 0.7869 / 0.7800 / 0.7885 |
| `clahe` | 0.7874 | 0.7930 | 0.7948 | 0.7409 | 0.7791 | 0.7818 / 0.7815 / 0.7739 |
| `enh3` | 0.7702 | 0.7957 | 0.8056 | 0.8037 | 0.7938 | 0.7954 / 0.7907 / 0.7953 |

`hybrid_weber` − `hybrid` by held-out dataset and seed:

| Seed | CHASE_DB1 | DRIVE | HRF | STARE |
|---|---|---|---|---|
| 1 | -0.0252 | +0.0062 | -0.0491 | +0.0017 |
| 2 | -0.0097 | -0.0048 | +0.0003 | +0.0090 |
| 3 | -0.0153 | -0.0087 | -0.0018 | +0.0084 |

## FIVES (8-fold TTA, 792 images)

| Variant | Mean | Seed 1 / 2 / 3 |
|---|---|---|
| `hybrid_weber` | 0.7815 | 0.7843 / 0.7709 / 0.7894 |
| `hybrid` | 0.7401 | 0.7306 / 0.7489 / 0.7409 |
| `rgb` | 0.7706 | 0.7547 / 0.7872 / 0.7699 |
| `clahe` | 0.7706 | 0.7575 / 0.7904 / 0.7638 |
| `enh3` | 0.7642 | 0.7669 / 0.7728 / 0.7528 |

## Degradation test: Dice lost from own original images (no TTA)

| Variant | disc_32 | fives |
|---|---|---|
| `hybrid_weber` (3 seeds) | -0.0140 | -0.0105 |
| `hybrid` | -0.0089 | -0.0253 |

## Decision (rule in PROTOCOL.md, stage 6b)

- C1, FIVES ≥ rgb mean and above rgb in ≥ 2 seeds: **yes**
- C2, LODO ≥ rgb mean: **yes**
- C3, robustness within 0.01 of hybrid: **yes**

**CONFIRMED: at least as good as rgb on LODO and FIVES, and robust; final model**
