# Stage 6: ablation of EnhancementV3's changes on hybrid (seed 1)

Reference `hybrid` seed 1: LODO 0.7983, FIVES 0.7306, degradation loss disc_32 -0.0106, fives -0.0248.

R1: LODO ≥ hybrid − 0.005. R2: loss at disc_32 and fives ≥ hybrid's − 0.01. R3 (exploratory): FIVES ≥ hybrid + 0.01.

| Variant | LODO | − hybrid | R1 | disc_32 loss | fives loss | R2 | FIVES | − hybrid | R3 | Reading |
|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_masknorm` | 0.7952 | -0.0031 | yes | -0.0023 | -0.0132 | yes | 0.7042 | -0.0263 | no | harmless, no FIVES effect |
| `hybrid_rg` | 0.7737 | -0.0246 | no | -0.0129 | -0.0429 | no | 0.7644 | +0.0338 | yes | helps FIVES at the cost of robustness |
| `hybrid_floor` | 0.7949 | -0.0034 | yes | -0.0108 | -0.0230 | yes | 0.7197 | -0.0109 | no | harmless, no FIVES effect |
| `hybrid_weber` | 0.7817 | -0.0166 | no | -0.0071 | -0.0041 | yes | 0.7843 | +0.0537 | yes | rejected |

Seed 1 is screening only: no variant is reported as better or worse than hybrid from these runs alone.
