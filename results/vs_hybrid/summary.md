# Results (dice)

Runs folder: `runs` · reference for paired tests: `hybrid`

## Leave-one-dataset-out (each column = a dataset never seen in training)

| variant | seeds | CHASE_DB1 | DRIVE | HRF | STARE | **mean** | clDice | AUC | precision | sensitivity |
|---|---|---|---|---|---|---|---|---|---|---|
| clahe | 1 | 0.7873 | 0.7970 | 0.7877 | 0.7553 | **0.7818** | 0.8124 | 0.9773 | 0.7640 | 0.8240 |
| clahe_scale_strong | 1 | 0.7381 | 0.7829 | 0.7369 | 0.8059 | **0.7660** | 0.8237 | 0.9803 | 0.6887 | 0.8753 |
| enh2c_scale_strong | 1 | 0.7534 | 0.7747 | 0.7378 | 0.8071 | **0.7682** | 0.8293 | 0.9795 | 0.6989 | 0.8670 |
| hybrid | 1 | 0.8068 | 0.7856 | 0.8092 | 0.7915 | **0.7983** | 0.8225 | 0.9825 | 0.7763 | 0.8452 |
| hybrid_scale | 1 | 0.7619 | 0.7907 | 0.7575 | 0.7981 | **0.7771** | 0.8247 | 0.9803 | 0.7118 | 0.8674 |
| hybrid_scale_strong | 1 | 0.7396 | 0.7878 | 0.7401 | 0.8086 | **0.7690** | 0.8258 | 0.9807 | 0.6908 | 0.8777 |
| hybrid_scale_strong_fovnorm | 1 | 0.7494 | 0.7956 | 0.7265 | 0.8167 | **0.7721** | 0.8330 | 0.9814 | 0.6849 | 0.8843 |
| hybrid_strong | 1 | 0.7866 | 0.7962 | 0.8001 | 0.8020 | **0.7962** | 0.8271 | 0.9821 | 0.7517 | 0.8635 |
| rgb | 1 | 0.7889 | 0.8058 | 0.7802 | 0.7727 | **0.7869** | 0.8154 | 0.9751 | 0.7625 | 0.8268 |
| rgb_scale_strong | 1 | 0.7650 | 0.7932 | 0.7300 | 0.8216 | **0.7775** | 0.8256 | 0.9784 | 0.7027 | 0.8655 |

Paired comparison with `hybrid` (per-image dice, averaged over seeds):

| variant | images | mean difference | images better / worse | Wilcoxon p |
|---|---|---|---|---|
| clahe | 113 | -0.0178 | 25 / 88 | 5.7e-13 |
| clahe_scale_strong | 113 | -0.0437 | 20 / 93 | 2e-14 |
| enh2c_scale_strong | 113 | -0.0409 | 17 / 96 | 4.4e-15 |
| hybrid_scale | 113 | -0.0296 | 28 / 85 | 8.5e-13 |
| hybrid_scale_strong | 113 | -0.0407 | 29 / 84 | 5.5e-13 |
| hybrid_scale_strong_fovnorm | 113 | -0.0409 | 37 / 76 | 1.7e-11 |
| hybrid_strong | 113 | -0.0049 | 35 / 78 | 0.0006 |
| rgb | 113 | -0.0157 | 39 / 74 | 0.00063 |
| rgb_scale_strong | 113 | -0.0352 | 36 / 77 | 4.9e-10 |

Notes: metrics are computed inside the FOV at the original resolution with 8-fold TTA.
With several variants compared, treat p-values above ~0.01 as inconclusive (multiple comparisons).