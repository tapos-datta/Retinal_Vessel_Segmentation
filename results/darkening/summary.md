# Darkening test

LODO models on their own held-out images, no TTA, threshold 0.5. Variants: hybrid, rgb, clahe; seeds: 3. Level = target mean green inside the FOV.

## Mean Dice by brightness level

| Level | darkened | hybrid | rgb | clahe | hybrid − rgb | better / worse | p | hybrid − clahe | better / worse | p |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 0 | 0.7955 | 0.7746 | 0.7779 | +0.0209 | 75 / 38 | 3e-06 | +0.0176 | 96 / 17 | 2e-17 |
| gain_46 | 106 | 0.7952 | 0.7809 | 0.7813 | +0.0142 | 82 / 31 | 3.4e-07 | +0.0139 | 90 / 23 | 5.6e-10 |
| gain_38 | 113 | 0.7951 | 0.7796 | 0.7796 | +0.0155 | 91 / 22 | 4.4e-09 | +0.0155 | 96 / 17 | 7.6e-11 |
| gain_25 | 113 | 0.7948 | 0.7635 | 0.7690 | +0.0313 | 101 / 12 | 1.5e-14 | +0.0259 | 104 / 9 | 2.5e-15 |
| gain_15 | 113 | 0.7941 | 0.6900 | 0.7334 | +0.1041 | 109 / 4 | 1.1e-19 | +0.0607 | 110 / 3 | 9.3e-20 |
| uneven_46 | 109 | 0.7919 | 0.7765 | 0.7780 | +0.0155 | 83 / 30 | 2.2e-07 | +0.0140 | 92 / 21 | 7.9e-12 |
| uneven_38 | 113 | 0.7918 | 0.7731 | 0.7749 | +0.0187 | 92 / 21 | 2.9e-09 | +0.0170 | 101 / 12 | 7.7e-14 |
| uneven_25 | 113 | 0.7923 | 0.7525 | 0.7615 | +0.0398 | 102 / 11 | 3.9e-15 | +0.0308 | 107 / 6 | 1e-16 |
| uneven_15 | 113 | 0.7889 | 0.6700 | 0.7138 | +0.1188 | 110 / 3 | 3.5e-20 | +0.0751 | 111 / 2 | 4.8e-20 |

## hybrid − rgb, by held-out dataset

| Level | CHASE_DB1 | DRIVE | HRF | STARE |
|---|---|---|---|---|
| original | +0.0166 (23/5) | -0.0141 (1/19) | +0.0389 (45/0) | +0.0211 (6/14) |
| gain_46 | +0.0150 (23/5) | +0.0025 (12/8) | +0.0342 (44/1) | -0.0199 (3/17) |
| gain_38 | +0.0143 (21/7) | +0.0094 (18/2) | +0.0318 (44/1) | -0.0135 (8/12) |
| gain_25 | +0.0364 (26/2) | +0.0318 (20/0) | +0.0352 (44/1) | +0.0150 (11/9) |
| gain_15 | +0.1800 (28/0) | +0.1133 (20/0) | +0.0515 (44/1) | +0.1070 (17/3) |
| uneven_46 | +0.0192 (24/4) | +0.0034 (12/8) | +0.0362 (44/1) | -0.0243 (3/17) |
| uneven_38 | +0.0245 (24/4) | +0.0115 (18/2) | +0.0350 (44/1) | -0.0188 (6/14) |
| uneven_25 | +0.0576 (28/0) | +0.0447 (20/0) | +0.0380 (44/1) | +0.0141 (10/10) |
| uneven_15 | +0.1906 (28/0) | +0.1569 (20/0) | +0.0623 (44/1) | +0.1075 (18/2) |

## hybrid − rgb, by seed (same-seed means)

| Level | seed1 | seed2 | seed3 |
|---|---|---|---|
| original | +0.0207 | +0.0291 | +0.0128 |
| gain_46 | +0.0145 | +0.0215 | +0.0068 |
| gain_38 | +0.0150 | +0.0233 | +0.0081 |
| gain_25 | +0.0249 | +0.0495 | +0.0196 |
| gain_15 | +0.0724 | +0.1686 | +0.0713 |
| uneven_46 | +0.0139 | +0.0256 | +0.0068 |
| uneven_38 | +0.0157 | +0.0307 | +0.0096 |
| uneven_25 | +0.0295 | +0.0637 | +0.0262 |
| uneven_15 | +0.0875 | +0.1822 | +0.0868 |

## Decision

**gain**: hybrid − rgb +0.0209, +0.0142, +0.0155, +0.0313, +0.1041; falls at every step: no; negative in every seed at level 38 and darker: no. **Rule: NOT SUPPORTED**

**uneven**: hybrid − rgb +0.0209, +0.0155, +0.0187, +0.0398, +0.1188; falls at every step: no; negative in every seed at level 38 and darker: no. **Rule: NOT SUPPORTED**


Paired tests use per-image Dice averaged over seeds; they treat images, not seeds, as the sample. Neither kind of darkening imitates blur, colour casts or reduced vessel contrast.
