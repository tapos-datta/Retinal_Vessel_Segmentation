# Degradation test

Frozen LODO models on their own held-out images, no TTA, threshold 0.5. Mean over the 113 images, averaged over seeds. Rule in PROTOCOL.md ("Degradation test").

## Calibration: measured image properties

Median over images. Left: absolute values; right: value / the same image's original value.

| Level | brightness | bright_outlier | vessel_contrast | stretched_contrast | noise | sharpness | brightness × | bright_outlier × | vessel_contrast × | noise × | sharpness × |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *training → FIVES darkest quarter* | *59.8 → 32.01* | *2.531 → 5.484* | *0.1257 → 0.1071* | *0.0801 → 0.0624* | *0.0164 → 0.0181* | *0.59 → 0.861* | *0.54* | *2.17* | *0.85* | *1.10* | *1.46* |
| original | 62.37 | 2.525 | 0.125 | 0.07934 | 0.01623 | 0.5955 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| disc_38 | 38.4 | 5.381 | 0.2222 | 0.07724 | 0.02956 | 0.6387 | 0.62 | 2.13 | 1.69 | 1.65 | 1.07 |
| disc_32 | 32.52 | 5.419 | 0.2188 | 0.07659 | 0.02999 | 0.6408 | 0.52 | 2.12 | 1.70 | 1.69 | 1.05 |
| disc_25 | 25.62 | 5.37 | 0.2105 | 0.07744 | 0.03094 | 0.6418 | 0.41 | 2.11 | 1.69 | 1.81 | 1.06 |
| contrast_85 | 62.19 | 2.343 | 0.1053 | 0.07054 | 0.01544 | 0.5677 | 1.00 | 0.94 | 0.86 | 0.92 | 1.00 |
| contrast_70 | 62.12 | 2.216 | 0.08929 | 0.06204 | 0.01386 | 0.5665 | 1.00 | 0.87 | 0.70 | 0.82 | 1.04 |
| noise_110 | 62.38 | 2.525 | 0.125 | 0.07878 | 0.01796 | 0.6035 | 1.00 | 1.00 | 1.00 | 1.11 | 1.01 |
| noise_150 | 62.39 | 2.5 | 0.125 | 0.07666 | 0.02451 | 0.6085 | 1.00 | 1.00 | 1.00 | 1.51 | 1.04 |
| noise_200 | 62.4 | 2.536 | 0.1266 | 0.07524 | 0.0323 | 0.6661 | 1.00 | 1.00 | 1.00 | 2.00 | 1.09 |
| fives | 32.42 | 5.406 | 0.1111 | 0.05381 | 0.02104 | 0.7418 | 0.52 | 2.07 | 0.89 | 1.24 | 1.31 |
| strong | 25.56 | 5.227 | 0.09524 | 0.05118 | 0.03079 | 0.8131 | 0.41 | 2.08 | 0.77 | 1.52 | 1.45 |

Images whose tone change reached a limit (power curve 2.5, haze 0.6) and so match the targets only partly: disc_38 9, disc_32 9, disc_25 9, contrast_70 1, fives 52, strong 54

Rows of variants not run here come from `results/degradation`, limited to the same seeds, images and levels.

## Mean Dice by level

| Level | hybrid | rgb | clahe | enh3 | hybrid_weber | hybrid − rgb | better / worse | p | enh3 − rgb |
|---|---|---|---|---|---|---|---|---|---|
| original | 0.7955 | 0.7746 | 0.7779 | 0.7918 | 0.7870 | +0.0209 | 75 / 38 | 3e-06 | +0.0172 |
| disc_38 | 0.7872 | 0.7323 | 0.7694 | 0.7266 | 0.7727 | +0.0549 | 98 / 15 | 3.3e-11 | -0.0057 |
| disc_32 | 0.7866 | 0.7353 | 0.7653 | 0.7371 | 0.7730 | +0.0513 | 98 / 15 | 1.3e-10 | +0.0019 |
| disc_25 | 0.7855 | 0.7344 | 0.7599 | 0.7501 | 0.7738 | +0.0511 | 97 / 16 | 4.7e-11 | +0.0157 |
| contrast_85 | 0.7919 | 0.7579 | 0.7735 | 0.7761 | 0.7897 | +0.0341 | 99 / 14 | 4.3e-14 | +0.0183 |
| contrast_70 | 0.7889 | 0.7363 | 0.7700 | 0.7676 | 0.7845 | +0.0526 | 108 / 5 | 4.2e-19 | +0.0313 |
| noise_110 | 0.7951 | 0.7744 | 0.7774 | 0.7918 | 0.7872 | +0.0208 | 75 / 38 | 3.2e-06 | +0.0174 |
| noise_150 | 0.7937 | 0.7743 | 0.7751 | 0.7914 | 0.7873 | +0.0194 | 74 / 39 | 8e-06 | +0.0171 |
| noise_200 | 0.7917 | 0.7738 | 0.7727 | 0.7904 | 0.7866 | +0.0179 | 72 / 41 | 3.5e-05 | +0.0166 |
| fives | 0.7702 | 0.7351 | 0.7450 | 0.7612 | 0.7765 | +0.0351 | 97 / 16 | 2e-10 | +0.0261 |
| strong | 0.7617 | 0.7113 | 0.7264 | 0.7526 | 0.7734 | +0.0504 | 98 / 15 | 6.5e-13 | +0.0413 |

## hybrid − rgb: finding vessels vs drawing their width

| Level | Dice | cl_recall | width_ratio | sensitivity | precision | Dice by seed |
|---|---|---|---|---|---|---|
| original | +0.0209 | +0.0398 | -0.0106 | +0.0216 | +0.0230 | +0.0207, +0.0291, +0.0128 |
| disc_38 | +0.0549 | +0.0741 | +0.0462 | +0.1057 | +0.0084 | +0.0721, +0.0510, +0.0418 |
| disc_32 | +0.0513 | +0.0802 | +0.0581 | +0.1074 | +0.0006 | +0.0624, +0.0517, +0.0399 |
| disc_25 | +0.0511 | +0.1000 | +0.0750 | +0.1173 | -0.0098 | +0.0491, +0.0640, +0.0402 |
| contrast_85 | +0.0341 | +0.0579 | -0.0144 | +0.0303 | +0.0416 | +0.0348, +0.0375, +0.0299 |
| contrast_70 | +0.0526 | +0.0951 | -0.0221 | +0.0542 | +0.0544 | +0.0543, +0.0557, +0.0478 |
| noise_110 | +0.0208 | +0.0396 | -0.0106 | +0.0213 | +0.0231 | +0.0207, +0.0292, +0.0124 |
| noise_150 | +0.0194 | +0.0389 | -0.0106 | +0.0202 | +0.0219 | +0.0190, +0.0284, +0.0108 |
| noise_200 | +0.0179 | +0.0371 | -0.0110 | +0.0185 | +0.0210 | +0.0177, +0.0271, +0.0090 |
| fives | +0.0351 | +0.1223 | -0.0126 | +0.0899 | -0.0019 | +0.0269, +0.0549, +0.0235 |
| strong | +0.0504 | +0.1782 | -0.0097 | +0.1340 | -0.0182 | +0.0386, +0.0830, +0.0295 |

## hybrid − rgb Dice by held-out dataset

| Level | CHASE_DB1 | DRIVE | HRF | STARE |
|---|---|---|---|---|
| original | +0.0166 | -0.0141 | +0.0389 | +0.0211 |
| disc_38 | +0.1078 | +0.1238 | +0.0319 | -0.0360 |
| disc_32 | +0.1101 | +0.1108 | +0.0269 | -0.0355 |
| disc_25 | +0.1277 | +0.0941 | +0.0227 | -0.0349 |
| contrast_85 | +0.0147 | +0.0179 | +0.0322 | +0.0816 |
| contrast_70 | +0.0152 | +0.0210 | +0.0475 | +0.1480 |
| noise_110 | +0.0162 | -0.0141 | +0.0390 | +0.0209 |
| noise_150 | +0.0142 | -0.0143 | +0.0377 | +0.0189 |
| noise_200 | +0.0130 | -0.0147 | +0.0362 | +0.0162 |
| fives | +0.0600 | +0.0364 | +0.0493 | -0.0331 |
| strong | +0.0822 | +0.0513 | +0.0619 | -0.0211 |

## hybrid − rgb at `fives`, by calibration match (descriptive)

| Images | n | Dice | cl_recall | width_ratio |
|---|---|---|---|---|
| reached a limit (partial match) | 52 | +0.0270 | +0.1548 | -0.0199 |
| full match | 61 | +0.0420 | +0.0947 | -0.0063 |

## Stage 6: robustness of each variant against hybrid (rule R2)

Dice lost from the variant's own original images, against hybrid's loss on the same images and seeds. R2 holds if the loss is at most hybrid's + 0.01 at disc_32 and fives.

| Variant | Original | disc_38 loss (hybrid) | disc_32 loss (hybrid) | disc_25 loss (hybrid) | contrast_85 loss (hybrid) | contrast_70 loss (hybrid) | noise_110 loss (hybrid) | noise_150 loss (hybrid) | noise_200 loss (hybrid) | fives loss (hybrid) | strong loss (hybrid) | R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 0.7955 | -0.0083 (-0.0083) | -0.0089 (-0.0089) | -0.0100 (-0.0100) | -0.0035 (-0.0035) | -0.0066 (-0.0066) | -0.0003 (-0.0003) | -0.0018 (-0.0018) | -0.0038 (-0.0038) | -0.0253 (-0.0253) | -0.0338 (-0.0338) | – |
| hybrid_weber | 0.7870 | -0.0144 (-0.0083) | -0.0140 (-0.0089) | -0.0132 (-0.0100) | +0.0026 (-0.0035) | -0.0025 (-0.0066) | +0.0002 (-0.0003) | +0.0003 (-0.0018) | -0.0004 (-0.0038) | -0.0105 (-0.0253) | -0.0136 (-0.0338) | yes |

Paired tests use per-image values averaged over seeds; they treat images, not seeds, as the sample.
