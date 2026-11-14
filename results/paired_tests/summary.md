# Paired per-image tests (rule: PROTOCOL.md, fixed 2026-11-13 before computing)

Two-sided Wilcoxon signed-rank on per-image Dice, pairs = images, from the existing evaluate.py per_image.csv files (8-fold TTA, threshold 0.5, Dice inside the FOV at original resolution); no model was run. FIVES: 792 images with a non-blank label. LODO: the 113 pooled images, each scored by the model of the fold that held its dataset out. Difference = first model − second. Median CI: 95% distribution-free (binomial order statistics; actual coverage in results.json). Significance level 0.005 (stage 2's Bonferroni level); 16 tests are run here, and a strict Bonferroni level for 16 would be 0.0031, so p between the two is marked. Per-seed and 3-seed-mean analyses were both declared in advance; neither is the primary one.

## FIVES: `hybrid_weber` vs `rgb`

| analysis | n | median diff [95% CI] | mean diff | images favouring `hybrid_weber` / `rgb` / tied | Wilcoxon p | p < 0.005 |
|---|---|---|---|---|---|---|
| seed 1 | 792 | +0.0266 [+0.0245, +0.0285] | +0.0296 | 86.5% / 13.5% / 0.0% | 4.9e-84 | yes |
| seed 2 | 792 | -0.0086 [-0.0109, -0.0071] | -0.0162 | 32.2% / 67.7% / 0.1% | 5.6e-41 | yes |
| seed 3 | 792 | +0.0178 [+0.0160, +0.0196] | +0.0195 | 79.3% / 20.7% / 0.0% | 6.6e-60 | yes |
| 3-seed per-image mean | 792 | +0.0130 [+0.0114, +0.0143] | +0.0109 | 74.4% / 25.6% / 0.0% | 1.2e-38 | yes |

Sign of the median difference in seeds 1 / 2 / 3: + / − / + (not the same in all three seeds). Seeds with p < 0.005 point in opposite directions.

## FIVES: `hybrid` vs `rgb`

| analysis | n | median diff [95% CI] | mean diff | images favouring `hybrid` / `rgb` / tied | Wilcoxon p | p < 0.005 |
|---|---|---|---|---|---|---|
| seed 1 | 792 | -0.0201 [-0.0225, -0.0171] | -0.0241 | 27.5% / 72.5% / 0.0% | 3.3e-38 | yes |
| seed 2 | 792 | -0.0275 [-0.0309, -0.0242] | -0.0383 | 24.1% / 75.6% / 0.3% | 3.2e-65 | yes |
| seed 3 | 792 | -0.0285 [-0.0323, -0.0255] | -0.0290 | 21.5% / 78.5% / 0.0% | 3.6e-64 | yes |
| 3-seed per-image mean | 792 | -0.0259 [-0.0296, -0.0224] | -0.0305 | 23.9% / 76.1% / 0.0% | 1.4e-58 | yes |

Sign of the median difference in seeds 1 / 2 / 3: − / − / − (the same in all three seeds).

## LODO: `hybrid_weber` vs `rgb`

| analysis | n | median diff [95% CI] | mean diff | images favouring `hybrid_weber` / `rgb` / tied | Wilcoxon p | p < 0.005 |
|---|---|---|---|---|---|---|
| seed 1 | 113 | -0.0151 [-0.0208, -0.0108] | -0.0087 | 19.5% / 80.5% / 0.0% | 5.1e-09 | yes |
| seed 2 | 113 | +0.0066 [-0.0040, +0.0159] | +0.0183 | 56.6% / 43.4% / 0.0% | 0.0043 | yes (between 0.0031 and 0.005) |
| seed 3 | 113 | -0.0004 [-0.0065, +0.0086] | +0.0067 | 49.6% / 50.4% / 0.0% | 0.44 | no |
| 3-seed per-image mean | 113 | -0.0015 [-0.0087, +0.0061] | +0.0055 | 43.4% / 56.6% / 0.0% | 0.66 | no |

Sign of the median difference in seeds 1 / 2 / 3: − / + / − (not the same in all three seeds). Seeds with p < 0.005 point in opposite directions.

## LODO: `hybrid_weber` vs `hybrid`

| analysis | n | median diff [95% CI] | mean diff | images favouring `hybrid_weber` / `hybrid` / tied | Wilcoxon p | p < 0.005 |
|---|---|---|---|---|---|---|
| seed 1 | 113 | -0.0239 [-0.0279, -0.0194] | -0.0244 | 19.5% / 80.5% / 0.0% | 1.2e-13 | yes |
| seed 2 | 113 | -0.0029 [-0.0048, -0.0010] | -0.0016 | 38.1% / 61.9% / 0.0% | 0.0054 | no |
| seed 3 | 113 | -0.0080 [-0.0094, -0.0058] | -0.0046 | 26.5% / 73.5% / 0.0% | 3.4e-06 | yes |
| 3-seed per-image mean | 113 | -0.0122 [-0.0145, -0.0097] | -0.0102 | 14.2% / 85.8% / 0.0% | 6e-13 | yes |

Sign of the median difference in seeds 1 / 2 / 3: − / − / − (the same in all three seeds).

## What these tests do and do not show

- An image-level paired test treats each trained model as fixed. Its p-value measures whether the two given models differ on these images; it says nothing about training (seed) variability, which is what the seed spread in the headline chart (figures/report/results_lodo_vs_fives.png) shows. The per-seed rows show whether the result repeats across seeds; three seeds are too few to test seed variability formally.
- With n = 792 on FIVES, a very small difference reaches any significance level. Read the size of the difference (median and its interval, the fraction of images favouring each model) before p.
- The 3-seed per-image mean reduces seed noise in each image's Dice, but it is still a test over images, not over training runs.
- FIVES informed the choice of `hybrid_weber` (stage 6), so its FIVES comparisons are not independent evidence (PROTOCOL.md, stage 6b).
