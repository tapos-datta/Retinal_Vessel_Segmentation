# Results (dice)

Runs folder: `runs` · results: `external_FIVES` · columns: condition · reference for paired tests: `rgb`

## Trained on all 113 images, tested on `external_FIVES`

| variant | seeds | amd | diabetic_retinopathy | glaucoma | normal | **mean** | clDice | AUC | precision | sensitivity |
|---|---|---|---|---|---|---|---|---|---|---|
| clahe | 3 | 0.7942 ± 0.0140 | 0.7363 ± 0.0195 | 0.6987 ± 0.0215 | 0.8461 ± 0.0149 | **0.7686 ± 0.0174** | 0.8024 | nan | 0.9667 | 0.6579 |
| hybrid | 3 | 0.7954 ± 0.0104 | 0.7099 ± 0.0075 | 0.6326 ± 0.0160 | 0.8167 ± 0.0093 | **0.7382 ± 0.0091** | 0.7753 | nan | 0.9552 | 0.6196 |
| rgb | 3 | 0.7924 ± 0.0136 | 0.7402 ± 0.0191 | 0.7027 ± 0.0208 | 0.8400 ± 0.0118 | **0.7686 ± 0.0162** | 0.8159 | nan | 0.9607 | 0.6591 |

Paired comparison with `rgb` (per-image dice, averaged over seeds):

| variant | images | mean difference | images better / worse | Wilcoxon p |
|---|---|---|---|---|
| clahe | 794 | -0.0000 | 393 / 399 | 0.33 |
| hybrid | 794 | -0.0304 | 189 / 603 | 1.4e-58 |

Notes: metrics are computed inside the FOV at the original resolution with 8-fold TTA.
With several variants compared, treat p-values above ~0.01 as inconclusive (multiple comparisons).