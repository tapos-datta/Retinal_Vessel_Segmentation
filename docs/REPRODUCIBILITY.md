# Reproducibility Guide (`v2`)

This guide lists the exact commands to rebuild datasets, train models, run evaluations/diagnostics, and regenerate figures.

See [DATA.md](DATA.md) for where each dataset comes from and how it is prepared, [METHODS.md](METHODS.md) for the model, and [PROTOCOL.md](PROTOCOL.md) for the evaluation protocol and the decision rules each stage had to meet.

---

## 1. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Verify GPU/MPS availability, dataset paths, and iteration speed:

```bash
python check_env.py
```

---

## 2. Data Preparation & Splits

1. **Pool the 4 core datasets** (`HRF`, `CHASE_DB1`, `DRIVE`, `STARE` → `data/pool/` with PNGs, FOV masks, and `metadata.csv`):

   ```bash
   python prepare_data.py --src <path/to/core_datasets> --out data/pool
   python make_splits.py
   ```

   This generates `splits/lodo.json` (4 folds), `splits/cv5.json` (5 folds), and `splits/legacy.json` (the original `v1` 11-image split). Add `--official-fov <dir> <dir>` if using official DRIVE/HRF FOV masks.

2. **Prepare external FIVES** (792 images after deduplicating 6 exact duplicates and removing 2 blank masks):

   ```bash
   python prepare_data.py --layout fives --src <path/to/FIVES> --out data/external_FIVES
   python make_splits.py   # adds splits/all.json; lodo/cv5/legacy remain unchanged
   ```

---

## 3. Training & Evaluation

### Single Run
```bash
python train.py --config configs/variants/hybrid_weber.json --split splits/lodo.json --fold 2 --seed 1
```
Override any config parameter with `--set`, e.g., `--set train.iterations=6000 train.batch_size=6`.

### Experiment Plans (`run_experiments.py`)
Runs automatically skip completed checkpoints and resume if interrupted. Use `--dry-run` to preview any plan:

```bash
python run_experiments.py --plan smoke       # quick setup check
python run_experiments.py --plan stage1      # LODO screening (6 variants)
python run_experiments.py --plan stage1b     # controls + scale x augmentation ablations
python run_experiments.py --plan stage2      # 3-seed LODO + 2-seed cv5 (hybrid, rgb, clahe)
python run_experiments.py --plan stage3      # train on all 113 images + test on FIVES
python run_experiments.py --plan stage6      # EnhancementV3 ablation on hybrid (seed 1)
python run_experiments.py --plan stage6b     # hybrid_weber 3-seed confirmation (LODO + FIVES)
python run_experiments.py --plan legacy      # v1 11-image split comparison
```

### Evaluate a Checkpoint on External Data
```bash
python evaluate.py runs/hybrid_weber/all/all/seed1 --external data/external_FIVES
```

### Aggregate Results
```bash
python aggregate.py --ref rgb
python aggregate.py --ref rgb --test-dir external_FIVES --by condition --out results/stage3_vs_rgb
python stage6b_summary.py
```

---

## 4. Diagnostic & Statistical Scripts

| Script | Purpose | Output Directory |
| :--- | :--- | :--- |
| `degradation_test.py` | Calibrated FIVES-matched degradation test (`--calibrate-only` for calibration) | `results/degradation/` |
| `darkening_test.py` | Uniform & ramp darkening robustness test | `results/darkening/` |
| `label_style_test.py` | Centreline recall & matched vessel width ratio | `results/label_style/` |
| `paired_tests.py` | Two-sided Wilcoxon signed-rank tests on `per_image.csv` | `results/paired_tests/` |
| `fov_edge.py` | Two-player Shapley split of FOV boundary vs. interior Dice | `results/fov_edge/` |
| `where_weber_wins.py` | Exploratory image-property correlations for `hybrid_weber` vs. `rgb` | `results/where_weber_wins/` |
| `fives_factors.py` | 10 image-property measurements on FIVES | `results/fives_factors/` |
| `fives_brightness_test.py` | Brightness-equalized FIVES test (200 images) | `results/fives_brightness/` |

---

## 5. Figures

Every figure in [RESULTS.md](RESULTS.md) is produced by one of these. All of them read the stored
result files or run frozen models; none trains anything. Pass `--device cpu` where a script offers
it if the GPU is shared.

```bash
# What the front-ends do to the image: five columns, one row per dataset plus FIVES
python visualize_enhancement.py --final --device cpu            # -> figures/enhancement_final/
python visualize_enhancement.py --checkpoint runs/hybrid/lodo/lodo_HRF/seed1/best.pt --out figures/v2_hrf

# What each model segments (TP/FP/FN overlays)
python prediction_figure.py --device cpu                        # the gallery showcase images
python prediction_figure.py --typical --device cpu              # one typical image per dataset

# Charts built from the stored results
python results_charts.py        # LODO vs FIVES, the headline chart   -> figures/report/
python degradation_figure.py    # Dice at every degradation level     -> figures/report/
python label_style_figure.py    # labelled and predicted vessel width -> figures/report/
python dataset_gap_final.py     # background-intensity alignment      -> figures/dataset_gap_final/
python edge_profile.py --checkpoint <run>/best.pt --out figures/<name>
```

`prediction_figure.py` caches probability maps under `results/prediction_figure/cache/`, so the
first run is slow and later ones are quick; `fov_edge.py` caches in the same way. Both caches are
excluded from Git and are rebuilt on demand.

## 6. What Each Run Saves

Each run under `runs/<variant>/<split>/<fold>/seed<N>/` saves:
* `config.json`, `fold.json`, `env.json` — exact hyperparameters, image lists, and package versions
* `train_log.csv`, `val_log.csv` — iteration-level loss and validation Dice
* `best.pt`, `last.pt` — model weights (selected by validation Dice)
* `test/per_image.csv`, `test/summary.json` — FOV-masked metrics at native resolution with 8-fold TTA
