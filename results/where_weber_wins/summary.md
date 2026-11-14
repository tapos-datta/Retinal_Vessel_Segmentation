# EXPLORATORY: on which images does `hybrid_weber` beat `rgb`?

**Exploratory, not confirmatory.** Begun after seeing the paired-test result (results/paired_tests); the analysis was fixed in PROTOCOL.md before computing. No claim here is confirmatory; a pattern found here would need a new pre-registered test on new data. No model was run.

Difference = `hybrid_weber` − `rgb` per-image Dice (8-fold TTA, threshold 0.5). LODO: 113 images, each scored by the fold that held its dataset out. FIVES: 792 images with a non-blank label.

## 1. Distribution of the difference

### LODO

| analysis | n | median | mean | P10 | P20 | P30 | P40 | P50 | P60 | P70 | P80 | P90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed 1 | 113 | -0.0151 | -0.0087 | -0.045 | -0.030 | -0.026 | -0.021 | -0.015 | -0.011 | -0.007 | -0.001 | +0.019 |
| seed 2 | 113 | +0.0066 | +0.0183 | -0.024 | -0.015 | -0.009 | -0.004 | +0.007 | +0.016 | +0.029 | +0.041 | +0.055 |
| seed 3 | 113 | -0.0004 | +0.0067 | -0.028 | -0.020 | -0.013 | -0.007 | -0.000 | +0.009 | +0.017 | +0.026 | +0.035 |
| 3-seed mean | 113 | -0.0015 | +0.0055 | -0.026 | -0.019 | -0.014 | -0.009 | -0.001 | +0.006 | +0.011 | +0.019 | +0.031 |

| analysis | images gaining / losing | gain > 0.02 / loss > 0.02 | gain > 0.05 / loss > 0.05 | gain > 0.1 / loss > 0.1 | total gain / total loss | top 10% of gaining images: share of total gain |
|---|---|---|---|---|---|---|
| seed 1 | 22 / 91 | 12 / 48 | 3 / 11 | 3 / 0 | +1.26 / -2.24 | 75% |
| seed 2 | 64 / 49 | 44 / 15 | 18 / 6 | 3 / 0 | +3.08 / -1.01 | 46% |
| seed 3 | 56 / 57 | 31 / 23 | 5 / 6 | 4 / 0 | +2.06 / -1.29 | 50% |
| 3-seed mean | 49 / 64 | 22 / 21 | 5 / 6 | 3 / 0 | +1.86 / -1.24 | 56% |

### FIVES

| analysis | n | median | mean | P10 | P20 | P30 | P40 | P50 | P60 | P70 | P80 | P90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed 1 | 792 | +0.0266 | +0.0296 | -0.006 | +0.008 | +0.016 | +0.022 | +0.027 | +0.032 | +0.039 | +0.049 | +0.076 |
| seed 2 | 792 | -0.0086 | -0.0162 | -0.056 | -0.033 | -0.022 | -0.015 | -0.009 | -0.004 | +0.001 | +0.005 | +0.012 |
| seed 3 | 792 | +0.0178 | +0.0195 | -0.013 | -0.001 | +0.007 | +0.012 | +0.018 | +0.023 | +0.029 | +0.038 | +0.057 |
| 3-seed mean | 792 | +0.0130 | +0.0109 | -0.019 | -0.005 | +0.003 | +0.009 | +0.013 | +0.018 | +0.022 | +0.029 | +0.041 |

| analysis | images gaining / losing | gain > 0.02 / loss > 0.02 | gain > 0.05 / loss > 0.05 | gain > 0.1 / loss > 0.1 | total gain / total loss | top 10% of gaining images: share of total gain |
|---|---|---|---|---|---|---|
| seed 1 | 685 / 107 | 495 / 48 | 154 / 20 | 43 / 7 | +26.68 / -3.25 | 30% |
| seed 2 | 255 / 536 | 30 / 260 | 10 / 96 | 1 / 24 | +3.06 / -15.91 | 42% |
| seed 3 | 628 / 164 | 361 / 60 | 108 / 25 | 26 / 10 | +19.98 / -4.56 | 34% |
| 3-seed mean | 589 / 203 | 282 / 76 | 54 / 31 | 5 / 10 | +13.99 / -5.32 | 30% |

## 2. Association with image properties (3-seed per-image mean difference)

45 tests in all (42 Spearman + 3 Kruskal–Wallis). p-values are not corrected; Bonferroni reference 0.05 / 45 = 0.0011, for orientation only. ρ > 0: higher values of the property go with `hybrid_weber` doing better relative to `rgb`.

### Spearman ρ (p), by group

| property | CHASE_DB1 (n=28) | DRIVE (n=20) | HRF (n=45) | STARE (n=20) | LODO pooled (n=113) | FIVES (n=792) |
|---|---|---|---|---|---|---|
| brightness | -0.28 (0.15) | -0.20 (0.39) | +0.07 (0.65) | +0.10 (0.66) | **-0.38** (3.1e-05) | **+0.76** (5.7e-149) |
| bright_outlier | +0.32 (0.095) | +0.42 (0.068) | -0.34 (0.024) | -0.18 (0.45) | +0.13 (0.17) | **-0.71** (2.8e-120) |
| vessel_contrast | -0.45 (0.017) | +0.01 (0.96) | -0.29 (0.051) | -0.61 (0.0041) | -0.22 (0.017) | **-0.25** (5e-13) |
| noise | +0.38 (0.044) | +0.27 (0.26) | -0.04 (0.77) | -0.45 (0.048) | -0.24 (0.011) | **-0.60** (1.3e-78) |
| sharpness | **+0.60** (0.00083) | +0.29 (0.21) | +0.02 (0.9) | -0.35 (0.13) | **+0.45** (5.1e-07) | **-0.28** (1.4e-15) |
| fov_diameter | -0.22 (0.25) | +0.21 (0.36) | +0.36 (0.015) | +0.29 (0.22) | **+0.53** (1.6e-09) | +0.01 (0.7) |
| vessel_fraction_in_fov | +0.27 (0.17) | -0.49 (0.027) | -0.41 (0.0057) | -0.46 (0.04) | **-0.48** (6.7e-08) | +0.09 (0.0076) |

Bold: p < 0.0011 (Bonferroni over 45 tests).

### Categorical properties (median difference, mean difference, fraction of images won, by category)

| property | category | n | median | mean | images won | Kruskal–Wallis p |
|---|---|---|---|---|---|---|
| dataset, LODO pooled | CHASE_DB1 | 28 | -0.0015 | -0.0014 | 36% | 3.4e-06 (45 tests) |
| dataset, LODO pooled | DRIVE | 20 | -0.0137 | -0.0156 | 0% |  |
| dataset, LODO pooled | HRF | 45 | +0.0137 | +0.0114 | 73% |  |
| dataset, LODO pooled | STARE | 20 | -0.0241 | +0.0229 | 30% |  |
| condition, HRF | diabetic_retinopathy | 15 | +0.0233 | +0.0230 | 87% | 7.8e-05 (45 tests) |
| condition, HRF | glaucoma | 15 | +0.0167 | +0.0167 | 93% |  |
| condition, HRF | healthy | 15 | -0.0068 | -0.0056 | 40% |  |
| condition, FIVES | amd | 194 | +0.0218 | +0.0254 | 93% | 6.2e-40 (45 tests) |
| condition, FIVES | diabetic_retinopathy | 200 | +0.0159 | +0.0173 | 85% |  |
| condition, FIVES | glaucoma | 198 | -0.0045 | -0.0122 | 40% |  |
| condition, FIVES | normal | 200 | +0.0121 | +0.0135 | 80% |  |

Figure: scatter.png, the difference against the two properties with the largest mean |ρ| over LODO pooled and FIVES (brightness: 0.57; noise: 0.42).

## 3. Reading (exploratory)

**LODO: a few large gains, not many small ones.** The positive mean (+0.0055) against a negative median (−0.0015) comes from three STARE images: im0324 (+0.388), im0319 (+0.344) and im0004 (+0.205), each large in every seed. On them `rgb` partly fails (Dice 0.38–0.61 across seeds) while `hybrid_weber` stays at 0.78–0.80. Without those three images the mean is −0.0029 and the median −0.0016. Apart from them, gains and losses are small and balanced (22 images gain and 21 lose more than 0.02). Two of the three are among the brightest images in the pool (green mean 131–132, STARE median 103).

**LODO: no shared property within datasets.** No property passes the Bonferroni reference within a dataset except sharpness in CHASE_DB1 (ρ +0.60), and the signs disagree between datasets (noise +0.38 in CHASE_DB1, −0.45 in STARE; bright_outlier +0.42 in DRIVE, −0.34 in HRF). The strong pooled correlations (fov_diameter +0.53, vessel fraction −0.48, sharpness +0.45, brightness −0.38) follow the dataset: `hybrid_weber` wins on 73% of HRF images (large eyes) and on none of DRIVE's (Kruskal–Wallis p 3e-6), so they describe datasets, not images within a dataset. In HRF the gain is on the diseased eyes (diabetic retinopathy 87% won, glaucoma 93%) and not the healthy ones (40%).

**FIVES: many small gains, ordered by brightness.** 589 of 792 images gain, and the 10% largest gains supply only 30% of the total, so the FIVES mean is a broad shift, not a few outliers. Brightness is the strongest association in the whole analysis (ρ +0.76): `hybrid_weber` beats `rgb` on 95–98% of the brighter half and on 23% of the darkest quarter (green mean below 37.5). Its correlates point the same way (bright_outlier −0.71 and relative noise −0.60 are both higher in dark images), and glaucoma, the darkest condition (median green 35.7 against 46.2), is the only one it loses (40% won). (Added after seeing the numbers, descriptive only: the brightness ρ is positive in each seed, +0.61 / +0.47 / +0.80, so seed 2's overall loss shifts the whole curve down rather than reversing it.)

**Same pattern on LODO and FIVES? No.** On FIVES the advantage grows steadily with brightness. On LODO there is no such gradient within any dataset (brightness ρ −0.28 to +0.10), and the pooled sign is the opposite (−0.38, a dataset effect). The one point of contact is that the three large LODO wins include the two brightest pool images, which fits the FIVES direction, but three images cannot carry a pattern. All of this is exploratory: the brightness link is a hypothesis for a pre-registered test on new data, not a finding.

