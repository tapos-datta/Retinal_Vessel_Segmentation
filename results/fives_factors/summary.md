# FIVES factors (exploratory)

| Property | with hybrid − rgb | within condition | after removing brightness | with clahe − rgb | R² alone | R² with brightness | FIVES outside training range |
|---|---|---|---|---|---|---|---|
| brightness | +0.83 | +0.79 | +nan | +0.06 | 0.68 | 0.68 | 34% |
| stretched_contrast | +0.18 | +0.15 | +0.47 | +0.17 | 0.03 | 0.75 | 4% |
| vessel_contrast | -0.05 | -0.02 | +0.36 | +0.12 | 0.00 | 0.72 | 3% |
| bright_outlier | -0.65 | -0.57 | +0.12 | -0.08 | 0.42 | 0.69 | 19% |
| sharpness | -0.41 | -0.30 | -0.15 | -0.12 | 0.17 | 0.69 | 3% |
| noise | -0.56 | -0.48 | +0.18 | -0.07 | 0.31 | 0.69 | 12% |
| shading | -0.34 | -0.31 | +0.20 | -0.04 | 0.11 | 0.69 | 0% |
| red_green | -0.45 | -0.42 | -0.05 | +0.05 | 0.20 | 0.68 | 0% |
| blue_green | -0.13 | +0.00 | +0.07 | -0.17 | 0.02 | 0.68 | 8% |
| vessel_fraction | +0.31 | +0.31 | +0.47 | +0.20 | 0.10 | 0.75 | 7% |

R² on ranks: all properties 0.78; all except brightness 0.65; brightness alone 0.68; condition alone 0.31; all properties plus condition 0.78

| Condition | n | hybrid − rgb | brightness | stretched_contrast | vessel_contrast | bright_outlier | sharpness | noise |
|---|---|---|---|---|---|---|---|---|
| amd | 194 | +0.003 | 61.592 | 0.060 | 0.088 | 2.704 | 0.612 | 0.013 |
| diabetic_retinopathy | 200 | -0.030 | 48.275 | 0.056 | 0.077 | 3.177 | 0.773 | 0.015 |
| glaucoma | 198 | -0.071 | 35.652 | 0.057 | 0.107 | 5.116 | 0.831 | 0.017 |
| normal | 200 | -0.023 | 42.908 | 0.081 | 0.119 | 4.089 | 0.769 | 0.017 |

Training images (100), median: brightness 59.799, stretched_contrast 0.080, vessel_contrast 0.126, bright_outlier 2.531, sharpness 0.590, noise 0.016, shading 4.596, red_green 2.497, blue_green 0.447, vessel_fraction 0.097
FIVES (median): brightness 46.173, stretched_contrast 0.064, vessel_contrast 0.097, bright_outlier 3.614, sharpness 0.754, noise 0.016, shading 2.774, red_green 2.241, blue_green 0.407, vessel_fraction 0.103
