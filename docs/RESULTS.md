# Experimental Results & Diagnostic Findings (`v2`)

This document summarizes all confirmed and exploratory findings of the U²-Net-E `v2` study. All training losses and evaluation metrics are computed **strictly inside the FOV mask** at native image resolution, using the **fused output (`d0`)**, 8-fold TTA (unless noted), Gaussian sliding-window blending, and a fixed `0.5` probability threshold.

See also: [METHODS.md](METHODS.md) for the model and variants, [PROTOCOL.md](PROTOCOL.md) for the evaluation protocol and the pre-registered decision rules, [DATA.md](DATA.md) for dataset provenance and preparation, and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the commands that produce every number below.

---

## 1. Datasets & Evaluation Protocols

| Dataset | Images | Resolution (px) | Mean FOV (px) | Labels Used | Patient Grouping |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **HRF** | 45 (15 H, 15 DR, 15 G) | 3504 × 2336 | 3376 | Official | 1 image / subject |
| **CHASE_DB1** | 28 (14 children, L/R) | 999 × 960 | 918 | `1stHO` | L + R eyes kept together |
| **DRIVE** | 20 (training split) | 565 × 584 | 542 | `manual1` | 1 image / subject |
| **STARE** | 20 | 700 × 605 | 666 | Hoover (`.ah`) | 1 image / subject |
| **FIVES** *(External)* | 792 (194 AMD, 200 DR, 198 G, 200 N) | 2048 × 2048 | — | Official | 6 duplicates & 2 blank masks removed |

* **Protocols (`make_splits.py`, split seed 0):**
  * **LODO (4-fold Leave-One-Dataset-Out):** Train on 3 datasets (10% val from training side), test on the 1 held-out dataset (all 113 images tested unseen).
  * **5-fold CV (`cv5`):** Stratified by dataset and HRF condition (in-distribution benchmark).
  * **Legacy (`splits/legacy.json`):** The original `v1` 11-image multi-source split for continuity.
  * **External (`splits/all.json` → FIVES):** Train on all 113 pooled images, test once on 792 FIVES images.
* **HRF FOV note:** Estimated HRF FOV masks reach the top, bottom, and right image frames (extending slightly into the dark border on the right), affecting all models equally.

---

## 2. Main Benchmark vs. External Results (3 Seeds)

### A. Cross-Dataset (LODO, 3 Seeds) & In-Distribution (5-Fold CV, 2 Seeds)

| Variant | CHASE_DB1 | DRIVE | HRF | STARE | LODO Mean | `cv5` Mean (2 seeds) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hybrid`** *(original enhancement)* | **0.801** ± 0.006 | 0.793 ± 0.006 | **0.808** ± 0.001 | 0.785 ± 0.005 | **0.7969** | 0.822 ± 0.000 |
| **`enh3`** *(EnhancementV3)* | 0.770 | 0.796 | 0.806 | **0.804** | 0.7938 | — |
| **`hybrid_weber`** *(final model)* | 0.784 | 0.790 | 0.792 | 0.792 | 0.7894 | — |
| **`rgb`** *(no enhancement)* | 0.786 ± 0.003 | **0.806** ± 0.001 | 0.780 ± 0.008 | 0.769 ± 0.008 | 0.7851 | **0.824** ± 0.000 |
| **`clahe`** *(classic CLAHE)* | 0.787 ± 0.001 | 0.793 ± 0.004 | 0.795 ± 0.008 | 0.741 ± 0.017 | 0.7791 | — |

* **LODO takeaways:** `hybrid` beats `rgb` (+0.016 per-image mean, 72/41 images, Wilcoxon $p = 8.5\times 10^{-5}$) and `clahe` (+0.017, 97/16, $p = 1.1\times 10^{-16}$), cuts worst-case failures (images < 0.70 Dice: 2 vs. 7 for `rgb`), and reduces the `cv5` → LODO drop by ~1/3. In `cv5`, `hybrid` and `rgb` are practically tied (−0.002).
* **Other screening findings (1 seed):** FOV-diameter scale normalization (`hybrid_scale` vs. `hybrid`) hurt LODO (−0.030; HRF −0.052); strong domain randomization (`hybrid_strong`) hurt slightly overall (−0.005); `EnhancementV2` + consistency (`enh2c_scale_strong`) did not beat controls in the scale-normalized recipe.

### B. External Validation on FIVES (792 Images, 3 Seeds)

| Variant | AMD (194) | DR (200) | Glaucoma (198) | Normal (200) | **FIVES All (792)** | clDice | AUC | Precision | Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hybrid_weber`** | — | — | — | — | **0.7815** | — | — | — | — |
| **`rgb`** | 0.792 | 0.740 | **0.710** | 0.840 | 0.7706 ± 0.016 | **0.818** | **0.977** | 0.963 | **0.661** |
| **`clahe`** | 0.794 | 0.736 | 0.706 | **0.846** | 0.7706 ± 0.017 | 0.804 | 0.976 | **0.969** | 0.660 |
| **`enh3`** | — | — | — | — | 0.7642 | — | — | — | — |
| **`hybrid`** | **0.795** | 0.710 | 0.639 | 0.817 | 0.7401 ± 0.009 | 0.777 | 0.972 | 0.958 | 0.621 |

* **`hybrid` fails on FIVES:** Loses **−0.030 Dice** vs. both `rgb` (189/603 images, $p = 1.4\times 10^{-58}$) and `clahe` (171/620, $p = 1.2\times 10^{-60}$), with more severe failures (71 images < 0.50 Dice vs. 53 for `rgb`). The deficit is largest in glaucoma (−0.071) and tracks darkness (Spearman $\rho = 0.83$).

---

## 3. Diagnostic Findings: Why `hybrid` Lost & How `hybrid_weber` Works

### A. Calibrated Degradation Test (`degradation_test.py`, No TTA, 3 Seeds)
LODO models were tested on their 113 held-out images degraded to match the darkest FIVES quartile (brighter optic disc, lower vessel contrast, higher noise, darker green channel):

| Level | `hybrid` | `rgb` | `clahe` | `enh3` | `hybrid_weber` | `hybrid` − `rgb` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | 0.7955 | 0.7746 | 0.7779 | 0.7918 | 0.7870 | +0.021 |
| **`disc_32`** | **0.7866** | 0.7353 | 0.7653 | 0.7371 | 0.7730 | +0.051 |
| **`fives` (matched)** | 0.7702 | 0.7351 | 0.7450 | 0.7612 | **0.7765** | **+0.035** (97/16) |
| **`strong`** | 0.7617 | 0.7113 | 0.7264 | 0.7526 | **0.7734** | +0.050 |

Dice lost from each model's own original images, at the two levels the stage 6/6b rules use:

| | `hybrid` | `rgb` | `clahe` | `enh3` | `hybrid_weber` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`disc_32`** | **−0.0089** | −0.0393 | −0.0126 | −0.0547 | −0.0140 |
| **`fives`** | −0.0253 | −0.0395 | −0.0329 | −0.0306 | **−0.0105** |

* **Key takeaway:** Model ranking under synthetic degradation (`hybrid` > `enh3` > `clahe` > `rgb`) is the **exact reverse** of FIVES (`rgb` = `clahe` > `enh3` > `hybrid`). Brightness, bright optic disc, contrast loss, and noise do not explain `hybrid`'s FIVES failure.
* **`hybrid_weber` is the most resistant model at the two matched chains** (`fives` 0.7765 and `strong` 0.7734, both the highest of the five) but slightly behind `hybrid` in the bright-disc family — the two sides of rule C3.
* **An oddity without an explanation:** `enh3` *gains* Dice as the optic disc is brightened (0.7266 at `disc_38` → 0.7501 at `disc_25`), so a calibrated degradation does not always make an image harder.

### B. Labelling-Style Test (`label_style_test.py`, No TTA, 3 Seeds)
* **Shared labelling-convention effect:** Relative to FOV diameter, FIVES labels vessels 2.6× wider than HRF (`6.08` vs `2.30` $\times 10^{-3}$). Consequently, **every model** predicts FIVES vessels 31–35% thinner than labelled (`width_ratio` −0.308 to −0.348).
* **Why `hybrid` loses vs. how `hybrid_weber` gains on FIVES:**
  * `hybrid` genuinely misses vessels (`cl_recall` 0.683 vs. `rgb` 0.739).
  * `hybrid_weber` has the highest `width_ratio` (**0.724** vs. `rgb` 0.690) and pixel sensitivity (**0.680** vs. 0.664), while its `cl_recall` (**0.733**) stays slightly *below* `rgb` (0.739). Thus, `hybrid_weber` gains on FIVES primarily by predicting thicker strokes that better match FIVES's wider labels, rather than superior vessel detection.

### C. Stage 6 Ablation & Stage 6b Confirmation (`hybrid_weber`, 3 Seeds)
Replacing `hybrid`'s per-patch min-max with multi-scale Weber normalization (`hybrid_weber`) met all three pre-registered Stage 6b rules:
* **C1 (FIVES 3-seed mean ≥ `rgb`):** `0.7815` vs `0.7706` (above `rgb` in seeds 1 & 3, below in seed 2: `0.7709` vs `0.7872`) — **Met**.
* **C2 (LODO 3-seed mean ≥ `rgb`):** `0.7894` vs `0.7851` (below `rgb` in seed 1: `0.7817` vs `0.7869`) — **Met**.
* **C3 (Degradation robustness within +0.010 of `hybrid`):** `disc_32` −0.0140, `fives` −0.0105 — **Met**.

### D. Paired Per-Image Tests (`paired_tests.py`, Two-Sided Wilcoxon)

| Comparison | Seed 1 | Seed 2 | Seed 3 | **3-Seed Per-Image Mean** | Consistent Sign? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FIVES: `hybrid_weber` − `rgb`** | +0.0266 (86.5%) | **−0.0086 (32.2%)** | +0.0178 (79.3%) | +0.0130 (74.4%), $p = 10^{-38}$ | **No** |
| **FIVES: `hybrid` − `rgb`** | −0.0201 (27.5%) | −0.0275 (24.1%) | −0.0285 (21.5%) | −0.0259 (23.9%), $p = 10^{-58}$ | Yes |
| **LODO: `hybrid_weber` − `rgb`** | −0.0151 (19.5%) | +0.0066 (56.6%) | −0.0004 (49.6%) | **−0.0015 (43.4%), $p = 0.66$** | **No** |
| **LODO: `hybrid_weber` − `hybrid`** | −0.0239 (19.5%) | −0.0029 (38.1%) | −0.0080 (26.5%) | −0.0122 (14.2%), $p = 6\times 10^{-13}$ | Yes |

### E. FOV-Edge Diagnostic (`fov_edge.py`, Seed 1, 113 LODO Images)

The prediction figures show a band of false positives along the FOV boundary in `hybrid_weber`'s output, strongest on HRF. Each image's Dice difference was split exactly into a boundary band and the interior by a two-player Shapley decomposition (band = FOV pixels within 2% of the FOV diameter of the boundary).

| Test set | Total (`hybrid_weber` − `hybrid`) | From the band | From the interior | Band share |
| :--- | :---: | :---: | :---: | :---: |
| CHASE_DB1 | −0.0252 | −0.0031 | −0.0221 | 12% |
| DRIVE | +0.0062 | +0.0000 | +0.0062 | 0% |
| HRF | −0.0491 | −0.0104 | −0.0387 | 21% |
| STARE | +0.0017 | +0.0050 | −0.0033 | — |
| **All 113** | **−0.0244** | **−0.0040** | **−0.0204** | **16%** |

* **The boundary is not the explanation.** 84% of the loss is in the interior; on HRF, `hybrid_weber` is still 0.039 behind `hybrid` on interior pixels alone. The band's share depends on the width chosen (6% / 21% / 41% on HRF at f = 0.01 / 0.02 / 0.04).
* On HRF the effect comes from the **image frame**, not the circular edge of the eye: with the frame excluded, HRF's band contribution is exactly 0.000. This is how the HRF FOV-mask property in §1 was found.
* Seed 1 is the `hybrid_weber` seed with the unusually low HRF result, so this is the most favourable case for an edge explanation, and the edge still does not explain it.

### F. Exploratory: Where `hybrid_weber` Wins (`where_weber_wins.py`)

Begun after the paired tests; the list of image properties was fixed before computing and all 45 tests are reported. Nothing here is confirmatory.

* **LODO — three images carry the gain.** The positive per-image mean (+0.0055) against a negative median (−0.0015) comes from three STARE images: `im0324` (+0.388), `im0319` (+0.344), `im0004` (+0.205), on which `rgb` partly fails (Dice 0.38–0.61) while `hybrid_weber` holds 0.78–0.80. Without them the mean is −0.0029. By dataset, `hybrid_weber` − `rgb` is −0.0014 (CHASE_DB1), −0.0156 (DRIVE), +0.0114 (HRF), +0.0229 (STARE); **of the +0.0043 LODO difference, STARE alone contributes +0.0057**, so rule C2 was met on the strength of those three images rather than on a general advantage.
* **FIVES — a broad shift ordered by brightness.** 589 of 792 images gain and the largest 10% supply only 30% of the total gain. Brightness is the strongest association in the study (Spearman ρ = +0.76, positive in every seed). `hybrid_weber` beats `rgb` on 95–98% of the brighter half and on 23% of the darkest quarter; glaucoma, the darkest condition, is the only one it loses (40% won). **This reverses the premise of the investigation:** the work began because the module loses on *dark* FIVES images, and the dark-image deficit is still unresolved.
* No image property explains the wins *within* a LODO dataset; the strong pooled correlations track the dataset, not the image.

### G. Appearance Alignment Against External Accuracy (`dataset_gap_final.py`, Seed 1)

The module was designed to make the datasets look alike to the network. The dataset gap — the mean distance between the per-dataset medians of background intensity, divided by the interquartile range — was recomputed with the final checkpoints, with FIVES added.

| Representation | Gap, 4 training sets | Gap, with FIVES | FIVES Dice (3 seeds) |
| :--- | :---: | :---: | :---: |
| Green channel | 0.67 | 0.92 | — |
| CLAHE (green) | 0.56 | 0.74 | 0.7706 |
| `hybrid` | 0.44 | **0.33** | 0.7401 |
| `hybrid_weber` | **0.26** | 0.35 | 0.7815 |

* **Alignment ran against performance.** Under `hybrid` the FIVES background median sits at 0.364, between HRF's 0.345 and STARE's 0.369 — among the training datasets, the closest alignment of any representation — and `hybrid` is the worst model on FIVES. Under `hybrid_weber` FIVES sits clearly apart (−0.162 against −0.069 to +0.075) and it has the highest FIVES mean.
* On the four training datasets the module does what it was designed to do, and the Weber version does it best (0.26 against 0.44 for `hybrid`).
* The measure is only the median of background intensity, so the conclusion is that *this* alignment measure fails to predict external accuracy, not that appearance is irrelevant. Values were recomputed with the final checkpoints and a fresh pixel sample, so they differ slightly from earlier tables; compare variants within a column, not across columns.

---

## 4. Hypotheses Summary (H1–H10)

| # | Hypothesis | Verdict |
| :--- | :--- | :--- |
| **H1** | Learned enhancement (`hybrid`) beats `rgb` on unseen datasets (LODO) | **Supported on LODO** (+0.016, $p = 8.5\times 10^{-5}$); **reversed on FIVES** (−0.030) |
| **H2** | `hybrid` beats classic CLAHE | **Supported on LODO** (+0.017, $p = 1.1\times 10^{-16}$); **reversed on FIVES** (−0.030) |
| **H3** | Classic CLAHE helps over `rgb` | **Not supported** (LODO −0.001, $p = 0.87$; FIVES 0.000, $p = 0.33$) |
| **H4** | FOV-diameter scale normalization improves generalization | **Rejected** (−0.030; HRF −0.052) |
| **H5** | Domain-randomization augmentation improves generalization | **Rejected overall** (−0.005) |
| **H6** | `EnhancementV2` + consistency beats original in scale-normalized recipe | **Not supported** |
| **H7** | `hybrid` gain holds on external FIVES | **Rejected** (−0.030 vs. `rgb` and `clahe`, $p = 1.4\times 10^{-58}$) |
| **H8** | FIVES gap is a labelling-style effect | **Partly** (all models predict ~30% thinner; `hybrid` also misses vessels) |
| **H9** | Dark-FIVES image properties explain `hybrid`'s loss | **Rejected** (`hybrid` leads `rgb` +0.035 under calibrated degradation) |
| **H10** | `hybrid_weber` matches/exceeds `rgb` on LODO & FIVES and keeps robustness | **Met C1–C3 on 3-seed means**, but **qualified by paired tests** (LODO median −0.0015, $p = 0.66$; FIVES seed 2 reverses; FIVES used in design) |

---

## 5. Limitations

| Limitation | Effect | What was done |
| :--- | :--- | :--- |
| **One external dataset** | "External" means one clinic and one camera | stated with every claim; IOSTAR and DR HAGIS could not be obtained (see [DATA.md](DATA.md)) |
| **FIVES informed later designs** | FIVES results after stage 3 are not independent | labelled exploratory; the final design was confirmed under a rule written before the runs |
| **113 training images, 4 datasets** | limited diversity; DRIVE and STARE have 20 images each | LODO and cv5 test every image; paired tests |
| **`hybrid_weber`'s margins sit inside the seed spread** | its parity with `rgb` is not reproducible run to run | paired per-image tests run and reported; the claim is "removes the deficit", never "beats" |
| **One seed in stages 1, 1b and 6** | seed noise not captured by image-level *p* values | key comparisons repeated with 3 seeds (stages 2, 4b, 6b) |
| **Automatic FOV masks; HRF masks reach the image frame** | a little non-retina area is scored in all 45 HRF images | found by the FOV-edge diagnostic; affects every variant equally, and stated wherever an HRF number is quoted |
| **DRIVE test images come from its training split** | not comparable with published DRIVE results | stated; adding the official 20 test images is open work |
| **Threshold fixed at 0.5** | penalizes over- or under-confident variants | adaptive thresholds tested and rejected on LODO |
| **~10 comparisons on the same images** | some small *p* values could be chance | *p* > 0.01 treated as inconclusive; Bonferroni 0.005 for headline claims |
| **Exploratory analyses after the result** | §3F was begun after seeing the paired tests | property list fixed before computing; all 45 tests reported; treated as hypotheses for new data |
| **No clinical claim** | these are benchmark and public-dataset metrics | nothing here establishes clinical performance |

## 6. Key Figures

Each folder holds the figure, its caption, and the values behind it.

| Path | What it shows |
| :--- | :--- |
| `figures/report/results_lodo_vs_fives.png` | **Headline chart.** LODO and FIVES side by side, five variants in the same order; the orders of merit are reversed, and the seed dots show which margins survive |
| `figures/predictions/typical/` | What each model segments on a typical image per dataset, chosen by a rule fixed before drawing (TP green, FP red, FN blue) |
| `figures/enhancement_final/` | What `hybrid` and `hybrid_weber` do to the image, one row per dataset plus a dark FIVES image |
| `figures/report/degradation_robustness.png` | Dice at every degradation level for the five variants, with example degraded images |
| `figures/report/label_style_widths.png` | Labelled vessel width per dataset, and predicted divided by labelled width per variant |
| `figures/dataset_gap_final/` | Background-intensity distributions per dataset under four representations, FIVES included |
| `figures/edge_profile/` | Average intensity profile across vessel edges |
| `results/where_weber_wins/scatter.png` | Per-image `hybrid_weber` − `rgb` difference against the two strongest image properties |
