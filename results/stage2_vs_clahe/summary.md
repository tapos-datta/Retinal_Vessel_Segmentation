# Results (dice)

Runs folder: `runs` · reference for paired tests: `clahe`

## cv5: cross-validation, every image tested once

| variant | seeds | CHASE_DB1 | DRIVE | HRF | STARE | **mean** | clDice | AUC | precision | sensitivity |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 2 | 0.8346 ± 0.0001 | 0.8018 ± 0.0008 | 0.8146 ± 0.0004 | 0.8411 ± 0.0033 | **0.8220 ± 0.0003** | 0.8479 | 0.9873 | 0.7854 | 0.8697 |
| rgb | 2 | 0.8353 ± 0.0005 | 0.8105 ± 0.0011 | 0.8164 ± 0.0000 | 0.8392 ± 0.0006 | **0.8241 ± 0.0004** | 0.8499 | 0.9876 | 0.7846 | 0.8737 |

## Leave-one-dataset-out (each column = a dataset never seen in training)

| variant | seeds | CHASE_DB1 | DRIVE | HRF | STARE | **mean** | clDice | AUC | precision | sensitivity |
|---|---|---|---|---|---|---|---|---|---|---|
| clahe | 3 | 0.7874 ± 0.0008 | 0.7930 ± 0.0039 | 0.7948 ± 0.0079 | 0.7409 ± 0.0167 | **0.7791** | 0.8158 | 0.9790 | 0.7605 | 0.8306 |
| clahe_scale_strong | 1 | 0.7381 | 0.7829 | 0.7369 | 0.8059 | **0.7660** | 0.8237 | 0.9803 | 0.6887 | 0.8753 |
| enh2c_scale_strong | 1 | 0.7534 | 0.7747 | 0.7378 | 0.8071 | **0.7682** | 0.8293 | 0.9795 | 0.6989 | 0.8670 |
| hybrid | 3 | 0.8011 ± 0.0062 | 0.7925 ± 0.0060 | 0.8084 ± 0.0008 | 0.7854 ± 0.0053 | **0.7969** | 0.8235 | 0.9827 | 0.7705 | 0.8486 |
| hybrid_scale | 1 | 0.7619 | 0.7907 | 0.7575 | 0.7981 | **0.7771** | 0.8247 | 0.9803 | 0.7118 | 0.8674 |
| hybrid_scale_strong | 1 | 0.7396 | 0.7878 | 0.7401 | 0.8086 | **0.7690** | 0.8258 | 0.9807 | 0.6908 | 0.8777 |
| hybrid_scale_strong_fovnorm | 1 | 0.7494 | 0.7956 | 0.7265 | 0.8167 | **0.7721** | 0.8330 | 0.9814 | 0.6849 | 0.8843 |
| hybrid_strong | 1 | 0.7866 | 0.7962 | 0.8001 | 0.8020 | **0.7962** | 0.8271 | 0.9821 | 0.7517 | 0.8635 |
| rgb | 3 | 0.7857 ± 0.0031 | 0.8057 ± 0.0007 | 0.7801 ± 0.0082 | 0.7689 ± 0.0075 | **0.7851** | 0.8145 | 0.9744 | 0.7567 | 0.8283 |
| rgb_scale_strong | 1 | 0.7650 | 0.7932 | 0.7300 | 0.8216 | **0.7775** | 0.8256 | 0.9784 | 0.7027 | 0.8655 |

Paired comparison with `clahe` (per-image dice, averaged over seeds):

| variant | images | mean difference | images better / worse | Wilcoxon p |
|---|---|---|---|---|
| clahe_scale_strong | 113 | -0.0256 | 24 / 89 | 5.3e-09 |
| enh2c_scale_strong | 113 | -0.0227 | 23 / 90 | 1.4e-08 |
| hybrid | 113 | +0.0166 | 97 / 16 | 1.1e-16 |
| hybrid_scale | 113 | -0.0115 | 27 / 86 | 1.2e-06 |
| hybrid_scale_strong | 113 | -0.0226 | 26 / 87 | 4.2e-08 |
| hybrid_scale_strong_fovnorm | 113 | -0.0228 | 34 / 79 | 3.9e-06 |
| hybrid_strong | 113 | +0.0133 | 77 / 36 | 2e-06 |
| rgb | 113 | +0.0009 | 59 / 54 | 0.87 |
| rgb_scale_strong | 113 | -0.0171 | 29 / 84 | 8.7e-06 |

Notes: metrics are computed inside the FOV at the original resolution with 8-fold TTA.
With several variants compared, treat p-values above ~0.01 as inconclusive (multiple comparisons).