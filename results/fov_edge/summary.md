# FOV-edge diagnostic (seed 1, LODO test sets, 8-fold TTA, threshold 0.5)

Band = FOV pixels within f x FOV diameter of the FOV boundary (image frame counts as boundary unless 'noframe'). Main f = 0.02. Dice differences split exactly into band + interior (two-player Shapley).

## hybrid_weber - hybrid, band f = 0.02 (frame)

| dataset | n | band area | Dice hybrid | Dice hybrid_weber | total diff | from band | from interior | band share of diff | error share in band (hybrid / hybrid_weber) |
|---|---|---|---|---|---|---|---|---|---|
| CHASE_DB1 | 28 | 7.8% | 0.8068 | 0.7817 | -0.0252 | -0.0031 | -0.0221 | 12% | 6.4% / 7.3% |
| DRIVE | 20 | 7.8% | 0.7856 | 0.7918 | +0.0062 | +0.0000 | +0.0062 | 0% | 5.4% / 5.5% |
| HRF | 45 | 9.3% | 0.8092 | 0.7601 | -0.0491 | -0.0104 | -0.0387 | 21% | 5.4% / 9.5% |
| STARE | 20 | 8.4% | 0.7915 | 0.7932 | +0.0017 | +0.0050 | -0.0033 | 297% | 8.6% / 6.1% |
| all (113) | 113 | 8.5% | 0.8013 | 0.7769 | -0.0244 | -0.0040 | -0.0204 | 16% | 5.5% / 9.2% |

## hybrid_weber - hybrid, band f = 0.02 (noframe)

| dataset | n | band area | Dice hybrid | Dice hybrid_weber | total diff | from band | from interior | band share of diff | error share in band (hybrid / hybrid_weber) |
|---|---|---|---|---|---|---|---|---|---|
| CHASE_DB1 | 28 | 7.8% | 0.8068 | 0.7817 | -0.0252 | -0.0031 | -0.0221 | 12% | 6.4% / 7.3% |
| DRIVE | 20 | 7.7% | 0.7856 | 0.7918 | +0.0062 | +0.0000 | +0.0062 | 0% | 5.4% / 5.5% |
| HRF | 45 | 2.3% | 0.8092 | 0.7601 | -0.0491 | +0.0000 | -0.0491 | -0% | 1.8% / 1.4% |
| STARE | 20 | 6.0% | 0.7915 | 0.7932 | +0.0017 | +0.0019 | -0.0002 | 112% | 6.3% / 5.1% |
| all (113) | 113 | 5.3% | 0.8013 | 0.7769 | -0.0244 | -0.0004 | -0.0240 | 2% | 2.3% / 1.9% |

## Band contribution vs band width (frame), all pairs

| dataset | pair | band f=0.01 | band f=0.02 | band f=0.04 | total |
|---|---|---|---|---|---|
| CHASE_DB1 | hybrid_weber - hybrid | -0.0023 | -0.0031 | -0.0044 | -0.0252 |
| CHASE_DB1 | hybrid_weber - rgb | +0.0005 | +0.0004 | +0.0001 | -0.0073 |
| CHASE_DB1 | hybrid - rgb | +0.0028 | +0.0036 | +0.0046 | +0.0179 |
| DRIVE | hybrid_weber - hybrid | +0.0000 | +0.0000 | -0.0004 | +0.0062 |
| DRIVE | hybrid_weber - rgb | -0.0006 | -0.0013 | -0.0028 | -0.0140 |
| DRIVE | hybrid - rgb | -0.0006 | -0.0014 | -0.0025 | -0.0202 |
| HRF | hybrid_weber - hybrid | -0.0028 | -0.0104 | -0.0200 | -0.0491 |
| HRF | hybrid_weber - rgb | +0.0015 | -0.0018 | -0.0031 | -0.0201 |
| HRF | hybrid - rgb | +0.0044 | +0.0086 | +0.0170 | +0.0289 |
| STARE | hybrid_weber - hybrid | +0.0035 | +0.0050 | +0.0055 | +0.0017 |
| STARE | hybrid_weber - rgb | -0.0011 | -0.0014 | -0.0012 | +0.0204 |
| STARE | hybrid - rgb | -0.0048 | -0.0068 | -0.0073 | +0.0188 |
| all (113) | hybrid_weber - hybrid | -0.0011 | -0.0040 | -0.0081 | -0.0244 |
| all (113) | hybrid_weber - rgb | +0.0004 | -0.0011 | -0.0019 | -0.0087 |
| all (113) | hybrid - rgb | +0.0015 | +0.0028 | +0.0062 | +0.0157 |
