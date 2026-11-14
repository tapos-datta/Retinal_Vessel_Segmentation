# Results (dice)

Runs folder: `runs` · reference for paired tests: `rgb_scale_strong`

## Leave-one-dataset-out (each column = a dataset never seen in training)

| variant | seeds | CHASE_DB1 | DRIVE | HRF | STARE | **mean** | clDice | AUC | precision | sensitivity |
|---|---|---|---|---|---|---|---|---|---|---|
| clahe_scale_strong | 1 | 0.7381 | 0.7829 | 0.7369 | 0.8059 | **0.7660** | 0.8237 | 0.9803 | 0.6887 | 0.8753 |
| enh2c_scale_strong | 1 | 0.7534 | 0.7747 | 0.7378 | 0.8071 | **0.7682** | 0.8293 | 0.9795 | 0.6989 | 0.8670 |
| hybrid | 1 | 0.8068 | 0.7856 | 0.8092 | 0.7915 | **0.7983** | 0.8225 | 0.9825 | 0.7763 | 0.8452 |
| hybrid_scale_strong | 1 | 0.7396 | 0.7878 | 0.7401 | 0.8086 | **0.7690** | 0.8258 | 0.9807 | 0.6908 | 0.8777 |
| hybrid_scale_strong_fovnorm | 1 | 0.7494 | 0.7956 | 0.7265 | 0.8167 | **0.7721** | 0.8330 | 0.9814 | 0.6849 | 0.8843 |
| rgb_scale_strong | 1 | 0.7650 | 0.7932 | 0.7300 | 0.8216 | **0.7775** | 0.8256 | 0.9784 | 0.7027 | 0.8655 |

Paired comparison with `rgb_scale_strong` (per-image dice, averaged over seeds):

| variant | images | mean difference | images better / worse | Wilcoxon p |
|---|---|---|---|---|
| clahe_scale_strong | 113 | -0.0085 | 41 / 72 | 1.3e-05 |
| enh2c_scale_strong | 113 | -0.0056 | 42 / 71 | 0.00045 |
| hybrid | 113 | +0.0352 | 77 / 36 | 4.9e-10 |
| hybrid_scale_strong | 113 | -0.0055 | 51 / 62 | 0.0034 |
| hybrid_scale_strong_fovnorm | 113 | -0.0057 | 37 / 76 | 4.6e-06 |

Notes: metrics are computed inside the FOV at the original resolution with 8-fold TTA.
With several variants compared, treat p-values above ~0.01 as inconclusive (multiple comparisons).