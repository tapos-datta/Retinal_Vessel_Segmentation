# Labelling-style test

Frozen models on data they never trained on, threshold 0.5. LODO = mean of the 4 held-out dataset means; FIVES = mean over its images; both averaged over seeds.

| Variant | Seeds (LODO / FIVES) | Measure | LODO | FIVES | FIVES − LODO |
|---|---|---|---|---|---|
| hybrid | 3 / 3 | dice | 0.7923 | 0.7415 | -0.0508 |
| hybrid | 3 / 3 | sensitivity | 0.8494 | 0.6258 | -0.2236 |
| hybrid | 3 / 3 | precision | 0.7591 | 0.9490 | +0.1899 |
| hybrid | 3 / 3 | cl_recall | 0.8279 | 0.6832 | -0.1446 |
| hybrid | 3 / 3 | cl_precision | 0.8269 | 0.9338 | +0.1070 |
| hybrid | 3 / 3 | width_ratio | 1.0288 | 0.6806 | -0.3482 |
| rgb | 3 / 3 | dice | 0.7766 | 0.7698 | -0.0068 |
| rgb | 3 / 3 | sensitivity | 0.8233 | 0.6637 | -0.1596 |
| rgb | 3 / 3 | precision | 0.7487 | 0.9505 | +0.2019 |
| rgb | 3 / 3 | cl_recall | 0.7879 | 0.7392 | -0.0488 |
| rgb | 3 / 3 | cl_precision | 0.8458 | 0.9334 | +0.0876 |
| rgb | 3 / 3 | width_ratio | 1.0212 | 0.6900 | -0.3313 |
| clahe | 3 / 3 | dice | 0.7739 | 0.7708 | -0.0031 |
| clahe | 3 / 3 | sensitivity | 0.8360 | 0.6621 | -0.1739 |
| clahe | 3 / 3 | precision | 0.7422 | 0.9620 | +0.2198 |
| clahe | 3 / 3 | cl_recall | 0.8161 | 0.7115 | -0.1046 |
| clahe | 3 / 3 | cl_precision | 0.8244 | 0.9621 | +0.1377 |
| clahe | 3 / 3 | width_ratio | 1.0216 | 0.7141 | -0.3075 |
| enh3 | 3 / 3 | dice | 0.7905 | 0.7659 | -0.0246 |
| enh3 | 3 / 3 | sensitivity | 0.8307 | 0.6599 | -0.1707 |
| enh3 | 3 / 3 | precision | 0.7687 | 0.9550 | +0.1863 |
| enh3 | 3 / 3 | cl_recall | 0.8079 | 0.7239 | -0.0840 |
| enh3 | 3 / 3 | cl_precision | 0.8510 | 0.9427 | +0.0918 |
| enh3 | 3 / 3 | width_ratio | 1.0311 | 0.7077 | -0.3234 |
| hybrid_weber | 3 / 3 | dice | 0.7862 | 0.7809 | -0.0052 |
| hybrid_weber | 3 / 3 | sensitivity | 0.8443 | 0.6800 | -0.1643 |
| hybrid_weber | 3 / 3 | precision | 0.7531 | 0.9468 | +0.1937 |
| hybrid_weber | 3 / 3 | cl_recall | 0.8173 | 0.7326 | -0.0846 |
| hybrid_weber | 3 / 3 | cl_precision | 0.8346 | 0.9355 | +0.1009 |
| hybrid_weber | 3 / 3 | width_ratio | 1.0427 | 0.7241 | -0.3185 |

## Decision (rule in PROTOCOL.md)

| Variant | (a) vessels found, pixels missed | (b) predicted width thinner on FIVES |
|---|---|---|
| hybrid | no | yes |
| rgb | yes | yes |
| clahe | no | yes |
| enh3 | no | yes |
| hybrid_weber | no | yes |

**MIXED: labelling style explains part of the gap, or only for some models**

## Secondary: differences between models on FIVES

| Comparison | Dice | cl_recall | width_ratio |
|---|---|---|---|
| hybrid − rgb | -0.0283 | -0.0559 | -0.0093 |
| enh3 − rgb | -0.0039 | -0.0153 | +0.0178 |
| hybrid − clahe | -0.0293 | -0.0283 | -0.0335 |

## Label statistics by dataset

| Dataset | Images | Mean width (px) | Width / FOV diameter (x1000) | Vessel fraction |
|---|---|---|---|---|
| CHASE_DB1 | 28 | 5.06 | 5.52 | 0.1003 |
| DRIVE | 20 | 2.53 | 4.67 | 0.1250 |
| FIVES | 792 | 12.26 | 6.08 | 0.0962 |
| HRF | 45 | 7.78 | 2.30 | 0.0838 |
| STARE | 20 | 3.28 | 4.92 | 0.0923 |

Width is the local diameter at centreline pixels (2 x distance to background - 1). The decision uses only the frozen models' outputs; label statistics are descriptive.
