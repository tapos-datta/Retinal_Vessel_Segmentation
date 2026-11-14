# FIVES at equal brightness (exploratory)

200 images, 3 seeds, no TTA, threshold 0.5 unless stated. Quartile 1 is darkest.

## Mean by brightness quartile

| Model | Input | Measure | Q1 | Q2 | Q3 | Q4 | All |
|---|---|---|---|---|---|---|---|
| clahe | original | dice | 0.770 | 0.823 | 0.789 | 0.766 | 0.787 |
| clahe | original | sensitivity | 0.658 | 0.719 | 0.679 | 0.648 | 0.676 |
| clahe | original | precision | 0.962 | 0.972 | 0.971 | 0.973 | 0.970 |
| clahe | original | auc | 0.969 | 0.985 | 0.980 | 0.973 | 0.977 |
| clahe | original | dice_0.3 | 0.800 | 0.851 | 0.821 | 0.800 | 0.818 |
| clahe | norm46 | dice | 0.747 | 0.815 | 0.797 | 0.786 | 0.787 |
| clahe | norm46 | sensitivity | 0.627 | 0.708 | 0.691 | 0.676 | 0.675 |
| clahe | norm46 | precision | 0.965 | 0.974 | 0.969 | 0.968 | 0.969 |
| clahe | norm46 | auc | 0.972 | 0.985 | 0.980 | 0.971 | 0.977 |
| clahe | norm46 | dice_0.3 | 0.782 | 0.845 | 0.828 | 0.817 | 0.818 |
| hybrid | original | dice | 0.691 | 0.781 | 0.772 | 0.786 | 0.757 |
| hybrid | original | sensitivity | 0.564 | 0.659 | 0.655 | 0.676 | 0.638 |
| hybrid | original | precision | 0.932 | 0.977 | 0.973 | 0.967 | 0.962 |
| hybrid | original | auc | 0.959 | 0.980 | 0.977 | 0.975 | 0.973 |
| hybrid | original | dice_0.3 | 0.729 | 0.816 | 0.806 | 0.816 | 0.792 |
| hybrid | norm46 | dice | 0.690 | 0.778 | 0.770 | 0.785 | 0.756 |
| hybrid | norm46 | sensitivity | 0.562 | 0.655 | 0.653 | 0.675 | 0.636 |
| hybrid | norm46 | precision | 0.932 | 0.977 | 0.973 | 0.966 | 0.962 |
| hybrid | norm46 | auc | 0.958 | 0.978 | 0.976 | 0.975 | 0.972 |
| hybrid | norm46 | dice_0.3 | 0.727 | 0.812 | 0.803 | 0.816 | 0.790 |
| rgb | original | dice | 0.769 | 0.819 | 0.791 | 0.770 | 0.787 |
| rgb | original | sensitivity | 0.660 | 0.718 | 0.684 | 0.654 | 0.679 |
| rgb | original | precision | 0.954 | 0.966 | 0.963 | 0.961 | 0.961 |
| rgb | original | auc | 0.970 | 0.985 | 0.981 | 0.976 | 0.978 |
| rgb | original | dice_0.3 | 0.799 | 0.847 | 0.821 | 0.802 | 0.817 |
| rgb | norm46 | dice | 0.742 | 0.812 | 0.798 | 0.783 | 0.784 |
| rgb | norm46 | sensitivity | 0.630 | 0.708 | 0.693 | 0.673 | 0.676 |
| rgb | norm46 | precision | 0.939 | 0.966 | 0.961 | 0.962 | 0.957 |
| rgb | norm46 | auc | 0.966 | 0.983 | 0.981 | 0.977 | 0.977 |
| rgb | norm46 | dice_0.3 | 0.773 | 0.841 | 0.827 | 0.814 | 0.814 |

## hybrid − other

| Comparison | Input | Q1 | Q2 | Q3 | Q4 | All | better / worse | p | Spearman with brightness |
|---|---|---|---|---|---|---|---|---|---|
| hybrid − clahe (dice) | original | -0.078 | -0.042 | -0.017 | +0.019 | -0.030 | 44 / 156 | 7.8e-16 | +0.85 |
| hybrid − clahe (dice_0.3) | original | -0.071 | -0.035 | -0.015 | +0.016 | -0.027 | 45 / 155 | 7.8e-16 | +0.82 |
| hybrid − clahe (auc) | original | -0.010 | -0.005 | -0.002 | +0.003 | -0.004 | 55 / 145 | 3e-13 | +0.61 |
| hybrid − clahe (dice) | norm46 | -0.057 | -0.038 | -0.027 | -0.001 | -0.031 | 19 / 181 | 1.2e-25 | +0.63 |
| hybrid − clahe (dice_0.3) | norm46 | -0.055 | -0.033 | -0.024 | -0.001 | -0.028 | 19 / 181 | 1.2e-25 | +0.61 |
| hybrid − clahe (auc) | norm46 | -0.014 | -0.006 | -0.004 | +0.004 | -0.005 | 33 / 167 | 2.3e-17 | +0.63 |
| hybrid − rgb (dice) | original | -0.078 | -0.039 | -0.019 | +0.016 | -0.030 | 45 / 155 | 1.2e-15 | +0.79 |
| hybrid − rgb (dice_0.3) | original | -0.070 | -0.031 | -0.016 | +0.014 | -0.026 | 53 / 147 | 3.1e-14 | +0.76 |
| hybrid − rgb (auc) | original | -0.011 | -0.005 | -0.003 | -0.001 | -0.005 | 43 / 157 | 7.9e-18 | +0.43 |
| hybrid − rgb (dice) | norm46 | -0.053 | -0.035 | -0.027 | +0.003 | -0.028 | 30 / 170 | 2e-21 | +0.63 |
| hybrid − rgb (dice_0.3) | norm46 | -0.046 | -0.028 | -0.024 | +0.002 | -0.024 | 29 / 171 | 8.6e-21 | +0.58 |
| hybrid − rgb (auc) | norm46 | -0.008 | -0.005 | -0.005 | -0.002 | -0.005 | 28 / 171 | 1.2e-21 | +0.30 |

## hybrid − rgb by seed (Dice, threshold 0.5)

| Seed | original | norm46 |
|---|---|---|
| seed1 | -0.0258 | -0.0195 |
| seed2 | -0.0369 | -0.0366 |
| seed3 | -0.0266 | -0.0281 |
