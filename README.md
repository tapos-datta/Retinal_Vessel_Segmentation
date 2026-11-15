# U²-Net-E: Cross-Dataset & External Generalization for Retinal Vessel Segmentation (`v2`)

A controlled benchmark and diagnostic study evaluating whether a learnable image-enhancement front-end improves retinal vessel segmentation across unseen datasets and external clinical cohorts.

> **Looking for the original `v1` prototype or CoreML iOS app?**  
> The initial prototype (`epoch 3834`, 80/10/10 split) is preserved on the [`v1-legacy`](../../tree/v1-legacy) branch (tag `v1.0`), which also backs the [RetinalVesselSegDemo](https://github.com/tapos-datta/RetinalVesselSegDemo) iOS CoreML demo.

**Documentation:** [Methods](docs/METHODS.md) · [Protocol & pre-registered rules](docs/PROTOCOL.md) · [Results](docs/RESULTS.md) · [Data](docs/DATA.md) · [Weights](docs/WEIGHTS.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

---

## Key Findings

1. **Benchmark (LODO) generalization does not guarantee external generalization.**  
   On 4-fold Leave-One-Dataset-Out (LODO) testing across **HRF, CHASE_DB1, DRIVE, and STARE** (113 images, 3 seeds), the original learned enhancement (`hybrid`) improves Dice by **+0.016** over no enhancement (`rgb`) and resists calibrated synthetic degradations (+0.035 vs. `rgb`). On the external **FIVES** dataset (792 images), however, `hybrid` **loses −0.030 Dice** (`0.740` vs. `0.771` for both `rgb` and `clahe`), in every seed, on about 76% of the images.
2. **Why the external gap happens.**  
   Every model predicts FIVES vessels ~30% thinner than FIVES ground-truth labels (a shared labelling-convention effect), while `hybrid`'s extra deficit comes from missed vessels—not brightness, bright optic disc, contrast, or noise.
3. **Nothing available before the external test predicted it.**  
   Three substitutes were tried and all three pointed the wrong way: LODO testing, calibrated degradation (whose model ranking is the *reverse* of FIVES), and input-distribution alignment—under `hybrid` the FIVES background distribution sits *among* the training datasets, the closest alignment of any representation tested, and `hybrid` is still the worst model on FIVES.
4. **Multi-scale Weber normalization (`hybrid_weber`) closes the external deficit without reliably surpassing `rgb`.**  
   Replacing per-patch min-max with multi-scale Weber normalization raises FIVES Dice to **0.7815** (vs. `rgb` **0.7706**) and LODO Dice to **0.7894** (vs. `rgb` **0.7851**), meeting all pre-registered confirmation rules (C1–C3). Paired per-image tests qualify this: the FIVES direction reverses in seed 2 (`p = 6e-41`), and the 3-seed LODO per-image median difference vs. `rgb` is `-0.0015` (`p = 0.66`). Its LODO parity rests on three STARE images where `rgb` partly fails, its FIVES advantage grows with image brightness so the darkest quarter is still lost, and the mechanism is a **thicker predicted vessel rather than better detection**. Because FIVES informed the design, `hybrid_weber` requires validation on a fresh external cohort.

| Variant | LODO Mean (3 seeds) | External FIVES (3 seeds) | Calibrated Degradation (`fives` Δ) |
| :--- | :---: | :---: | :---: |
| **`hybrid`** *(original enhancement)* | **0.7969** | 0.7401 | −0.0253 |
| **`enh3`** *(EnhancementV3)* | 0.7938 | 0.7642 | −0.0306 |
| **`hybrid_weber`** *(final model)* | 0.7894 | **0.7815** | **−0.0105** |
| **`rgb`** *(no enhancement)* | 0.7851 | 0.7706 | −0.0395 |
| **`clahe`** *(classic CLAHE)* | 0.7791 | 0.7706 | −0.0329 |

*Full per-dataset tables, diagnostics, hypotheses (H1–H10), and limitations: see [docs/RESULTS.md](docs/RESULTS.md).*

---

## Research Progression: From `v1` Prototype to `v2` Controlled Study

`v1` demonstrated end-to-end U²-Net-E training, cross-dataset patch sampling, and real-time CoreML deployment. Before scaling the study, `v2` audited `v1` (`run1`/`run2`) and corrected the following methodological oversights:

| Topic | `v1` Prototype (`v1-legacy`) | Audit Finding | `v2` Standard (`main`) |
| :--- | :--- | :--- | :--- |
| **Test split framing** | Called the 11-image 80/10/10 split a *"cross-domain test set"* | Test images came from the same 4 training datasets; DRIVE images came from DRIVE's training split. | Renamed to **held-out multi-source split** (`splits/legacy.json`); unseen-domain testing uses **4-fold LODO** (113 images) and **external FIVES** (792 images). |
| **HRF & peak claims** | Highlighted `0.8649` on HRF (`15_h`) as *"pathological robustness"* | `15_h` is a single healthy image; pathological HRF cases in the same table scored `0.766`–`0.784`. | Reports full per-dataset means across all 45 HRF images (LODO HRF mean **0.807**). |
| **Metric scope** | Whole-image Accuracy (`0.9705`) and Specificity (`0.9768`) | Scoring black pixels outside the retina inflated Accuracy and Specificity. | All losses and metrics are computed **strictly inside the FOV mask** at native resolution. |
| **Inference & output** | Side output `d1` + per-image probability min-max rescaling before 0.5 threshold | Per-image stretching alters the effective operating threshold per image. | Uses **fused output (`d0`)** with 8-fold TTA, Gaussian sliding-window blending, and direct 0.5 thresholding. |
| **Enhancement claim** | *"Enhancement universally improves generalization"* | Holds on LODO and synthetic degradation, but reverses on external FIVES (`-0.030` Dice). | Evaluates both LODO and external FIVES, diagnoses failure modes, and tests `hybrid_weber` across 3 seeds. |

The U²-Net backbone and original enhancement module (`u2nete/models/`) are kept unchanged from `v1` so comparisons are exact; `EnhancementV2.py` and `EnhancementV3.py` implement the redesigned front-ends. Architecture details and equations: [docs/METHODS.md](docs/METHODS.md).

---

## Quickstart

```bash
# 1. Environment  (Python >= 3.10)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: make `import u2nete` work from any directory
pip install -e .

# 2. Prepare pooled datasets & splits (HRF, CHASE_DB1, DRIVE, STARE)
python prepare_data.py --src <path/to/core_datasets> --out data/pool
python make_splits.py

# 3. Train a single fold or run an experiment plan
python train.py --config configs/variants/hybrid_weber.json --split splits/lodo.json --fold 2 --seed 1
python run_experiments.py --plan stage6b --dry-run

# 4. Evaluate on external FIVES & aggregate results
python evaluate.py runs/hybrid_weber/all/all/seed1 --external data/external_FIVES
python aggregate.py --ref rgb
```

For full reproduction commands (all stages, diagnostics, and figure scripts), see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## Repository Layout

```
u2nete/              model code: U²-Net-lite backbone and the enhancement front-ends
configs/variants/    one JSON per variant; a variant is a config, not a code branch
splits/              LODO, cv5, legacy and all splits (fixed split seed 0)
data/pool/           metadata.csv only; images are not distributed
results/             per-image CSVs and summary tables for every test
figures/             generated figures with their captions and the values behind them
docs/                METHODS, PROTOCOL, RESULTS, DATA, REPRODUCIBILITY
*.py                 training, evaluation, diagnostics and figure scripts
```

What is **not** in this repository: raw images, trained weights, cached probability maps, and the
project's internal running notes (decision log, protocol source and manuscript drafts). The public
documentation in `docs/` is written from those internal files.

---

## Data & Pretrained Checkpoints

Raw images and `.pt` checkpoints are not in Git; only `data/pool/metadata.csv`, the two manifests, `splits/`, `configs/`, the per-run logs and the summary tables are tracked. Nothing below is needed to check a published number: every run folder keeps its `config.json`, `fold.json`, `env.json` and `test/per_image.csv`.

- **Weights — open download.** The 15 checkpoints behind every figure (seed 1 of `rgb`, `hybrid` and `hybrid_weber`, on the four LODO folds and the `all` split) are attached to the [latest release](https://github.com/tapos-datta/Retinal_Vessel_Segmentation/releases) as `retinal-vessel-seg-v2-weights-seed1.tar.gz` (66 MB). No request or approval is needed. They are for research use under the same restrictions as the training data, and the Apache 2.0 licence does not extend to them. Details: [docs/WEIGHTS.md](docs/WEIGHTS.md).
- **Datasets — rebuild them yourself.** Download DRIVE, HRF, CHASE_DB1, STARE and FIVES from their official distributions and run `prepare_data.py` + `make_splits.py` (sources and terms: [docs/DATA.md](docs/DATA.md); commands: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)). `data/pool/MANIFEST.sha256` and `data/external_FIVES/MANIFEST.sha256` confirm, file by file, that your rebuild matches the one used here.
- **Images cannot be redistributed**, including the converted copies in `data/pool`, because several of the datasets do not allow it. Open an issue if you get stuck on the preparation step.

---

## Limitations in Brief

- **One external dataset.** "External" means FIVES: one clinic, one camera. FIVES also informed the design of `EnhancementV3` and the stage 6 ablation, so FIVES results after stage 3 are exploratory and `hybrid_weber` has not been tested on a truly independent cohort.
- **Small training pool.** 113 images from four public datasets; DRIVE and STARE contribute 20 each.
- **Seed variability.** Three seeds support the pre-registered mean checks, but seed 2's FIVES reversal shows run-to-run variance is not negligible.
- **No clinical claim.** These are benchmark and public-dataset segmentation metrics. Nothing here establishes clinical performance.

The full list, with what was done about each, is in [docs/RESULTS.md](docs/RESULTS.md) §5.

---

## Citing and Licence

Code and documentation are released under the [Apache License 2.0](LICENSE); see
[NOTICE](NOTICE) for scope and attribution. The datasets are **not** covered by it; see
[docs/DATA.md](docs/DATA.md). Citation metadata is in [`CITATION.cff`](CITATION.cff).

<!--## Future Work (Part 2)-->
<!---->
<!--Inspired by the labelling-convention and domain-shift findings in `v2`, Part 2 explores **federated learning across datasets as simulated hospitals** (using Flower with FedAvg, FedProx, FedBN, and a local labelling-style head) to address inter-clinic annotation differences without centralizing patient data.-->
