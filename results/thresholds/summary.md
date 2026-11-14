# Image-adaptive thresholds

LODO models on their held-out images, 8-fold TTA. Images: 113. Seeds: 1. LODO mean = mean of the per-dataset means.

## hybrid

| Method | LODO Dice | difference from fixed_0.5 | better / worse | p | precision | sensitivity | CHASE_DB1 | DRIVE | HRF | STARE |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed_0.5 | 0.7983 | +0.0000 | 0 / 0 | nan | 0.774 | 0.844 | 0.807 | 0.786 | 0.809 | 0.791 |
| val_fixed | 0.7919 | -0.0064 | 76 / 37 | 0.18 | 0.794 | 0.818 | 0.811 | 0.749 | 0.809 | 0.799 |
| otsu_prob | 0.7962 | -0.0021 | 36 / 77 | 2.1e-05 | 0.763 | 0.853 | 0.802 | 0.790 | 0.809 | 0.784 |
| otsu_logit | 0.7102 | -0.0881 | 9 / 104 | 1.3e-18 | 0.580 | 0.946 | 0.690 | 0.780 | 0.695 | 0.675 |
| hysteresis | 0.7862 | -0.0120 | 32 / 81 | 4e-08 | 0.728 | 0.879 | 0.784 | 0.800 | 0.805 | 0.756 |

## rgb

| Method | LODO Dice | difference from fixed_0.5 | better / worse | p | precision | sensitivity | CHASE_DB1 | DRIVE | HRF | STARE |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed_0.5 | 0.7869 | +0.0000 | 0 / 0 | nan | 0.772 | 0.822 | 0.789 | 0.806 | 0.780 | 0.773 |
| val_fixed | 0.7843 | -0.0026 | 82 / 31 | 0.004 | 0.801 | 0.791 | 0.794 | 0.783 | 0.787 | 0.773 |
| otsu_prob | 0.7841 | -0.0028 | 21 / 92 | 2.4e-12 | 0.758 | 0.831 | 0.786 | 0.807 | 0.771 | 0.773 |
| otsu_logit | 0.6873 | -0.0996 | 3 / 110 | 6.5e-19 | 0.553 | 0.930 | 0.682 | 0.768 | 0.626 | 0.674 |
| hysteresis | 0.7689 | -0.0181 | 14 / 99 | 1.2e-15 | 0.711 | 0.858 | 0.772 | 0.806 | 0.736 | 0.762 |

## clahe

| Method | LODO Dice | difference from fixed_0.5 | better / worse | p | precision | sensitivity | CHASE_DB1 | DRIVE | HRF | STARE |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed_0.5 | 0.7818 | +0.0000 | 0 / 0 | nan | 0.753 | 0.835 | 0.787 | 0.797 | 0.788 | 0.755 |
| val_fixed | 0.7774 | -0.0045 | 46 / 22 | 0.83 | 0.777 | 0.805 | 0.794 | 0.763 | 0.788 | 0.765 |
| otsu_prob | 0.7794 | -0.0024 | 62 / 51 | 0.12 | 0.742 | 0.846 | 0.783 | 0.799 | 0.791 | 0.745 |
| otsu_logit | 0.6525 | -0.1293 | 5 / 108 | 8.8e-20 | 0.512 | 0.949 | 0.681 | 0.777 | 0.526 | 0.626 |
| hysteresis | 0.7696 | -0.0122 | 50 / 63 | 0.001 | 0.705 | 0.874 | 0.764 | 0.805 | 0.795 | 0.715 |

## enh3

| Method | LODO Dice | difference from fixed_0.5 | better / worse | p | precision | sensitivity | CHASE_DB1 | DRIVE | HRF | STARE |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed_0.5 | 0.7954 | +0.0000 | 0 / 0 | nan | 0.775 | 0.832 | 0.768 | 0.802 | 0.803 | 0.809 |
| val_fixed | 0.7871 | -0.0083 | 29 / 64 | 5.9e-06 | 0.797 | 0.799 | 0.770 | 0.770 | 0.799 | 0.809 |
| otsu_prob | 0.7943 | -0.0011 | 55 / 57 | 0.22 | 0.765 | 0.841 | 0.766 | 0.804 | 0.805 | 0.803 |
| otsu_logit | 0.7384 | -0.0571 | 16 / 97 | 7.2e-17 | 0.618 | 0.936 | 0.691 | 0.781 | 0.771 | 0.711 |
| hysteresis | 0.7875 | -0.0080 | 43 / 70 | 0.00098 | 0.732 | 0.869 | 0.754 | 0.808 | 0.807 | 0.781 |

Precision and sensitivity are LODO means. Paired tests use per-image Dice and treat images as the sample.
