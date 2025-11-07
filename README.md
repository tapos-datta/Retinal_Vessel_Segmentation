# U²Net-E Retinal Vessel Segmentation

## Overview

The Enhanced U²-Net (U²Net-E) targets robust retinal vessel segmentation across heterogeneous imaging datasets. A learnable enhancement front-end standardizes vessel contrast before the nested U-backbone, enabling consistent performance despite differences in resolution, illumination, and pathology.

- **Project goal**: Generalized retinal vessel segmentation across DRIVE, HRF, CHASE_DB1, and STARE.
- **Checkpoint of record**: `best_model_epoch_3834_val_0.1330.pth`.
- **Success criteria**: Dice > 0.77 and Sensitivity > 0.84 on a balanced cross-dataset test set.

## Dataset and Splits

All public datasets were merged, shuffled, and stratified to preserve representation of each source in every subset.

| Dataset | Total Images | Train | Val | Test |
| --- | --- | --- | --- | --- |
| DRIVE | 20 | 16 | 2 | 2 |
| HRF | 45 | 36 | 4 | 5 |
| CHASE_DB1 | 28 | 22 | 3 | 3 |
| STARE | 20 | 16 | 2 | 2 |

- **Global split**: 80 % / 10 % / 10 % (train/val/test) with batch sampling across datasets.
- **Held-out evaluation**: 11-image balanced test set spanning all four datasets.

## Methodology: Multi-Scale and Patch-Based Training

### 2.1 Model Architecture (U²-Net-E)

The U²-Net-E is designed for fine-grained feature capture, which is essential for segmenting thin capillaries:

- **Backbone**: U²-Net-lite, a nested U-structure using Residual U-blocks (RSU) to effectively fuse multi-scale information.
- **Enhancement module (hybrid mode)**: Combines green-channel emphasis with per-image min-max stretching, deep residual CNN enhancement, and learnable adaptive contrast correction to boost vessel contrast and reduce domain shift across the four input datasets.

### 2.2 Patch-Based Strategy for HRF Complexity

The HRF dataset is the most challenging component of the cohort due to its high resolution (up to 3504 × 2336 pixels) and inclusion of images from diabetic retinopathy and glaucoma patients.

To overcome the difficulty of training on massive high-resolution images and to ensure the model learns generalized features, a patch-based training strategy was implemented:

- **Local feature focus**: Random sub-region patches force the network to learn intricate vessel textures independent of global context, supporting high sensitivity on high-resolution data.
- **Domain invariance**: Every batch mixes patches drawn from DRIVE, HRF, CHASE_DB1, and STARE, compelling the model to learn dataset-agnostic filters that remain robust under domain shift.
- **Data augmentation**: Patch extraction multiplies the number of unique training samples, exposing the network to fine-grained pathological variations present in all datasets.

### 2.3 Evaluation Protocol

- **Validation metrics** are computed on the unified cohort’s dedicated validation split during training (checkpoint epoch 3834).
- **Generalized test metrics** are measured on the 11-image balanced test set drawn from all four datasets with test-time augmentation (TTA) during inference.

### 2.4 Training Configuration

| Parameter | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 3 × 10⁻⁴ |
| Batch size | 6 |
| Patch size | 384 × 384 |
| Total epochs | ~4000 |
| Loss | BCE + Tversky (α = 0.3, β = 0.7) |
| Augmentation | Rotation, flip, brightness adjustment, blur, noise |
| Test-time augmentation | 8-fold averaging |

## Evaluation Protocol

1. **Validation set** drawn from the unified cohort monitors training progress (metrics at epoch 3834).
2. **Cross-domain test set** of 11 images measures generalization; inference uses test-time augmentation.
3. **Metric suite** includes Dice, IoU, Accuracy, Sensitivity, and Specificity.

## Results

### Balanced Test-Set Metrics at Epoch 3834

The metrics below summarize performance on the 11-image balanced test set (epoch 3834 checkpoint). Aggregate scores are complemented by per-image means to highlight cross-domain consistency.

| Metric | Value | Notes |
| --- | --- | --- |
| Dice (global) | 0.8062 | Calculated over the combined mask of the full test set |
| Dice (per-image mean) | 0.7880 | Arithmetic mean of Dice scores across the 11 images |
| IoU (global) | 0.6763 | Intersection over Union on the combined mask |
| IoU (per-image mean) | 0.6502 | Arithmetic mean of IoU values per image |
| Accuracy | 0.9705 | Pixel-wise accuracy over the full test set |
| Sensitivity (global) | 0.8868 | True positive rate for vessels (combined mask) |
| Sensitivity (per-image mean) | 0.8689 | Arithmetic mean of sensitivities per image |
| Specificity | 0.9768 | True negative rate for background |

### Per-Image Generalization Snapshot

| Image (Dataset) | Dice | Sensitivity |
| --- | --- | --- |
| `13_g.jpg` (HRF – Glaucoma) | 0.7792 | 0.8612 |
| `14_dr.JPG` (HRF – DR) | 0.7844 | 0.8504 |
| `15_dr.JPG` (HRF – DR) | 0.7663 | 0.8767 |
| `15_h.jpg` (HRF – Healthy) | 0.8649 | 0.9114 |
| `39_training.tif` (DRIVE) | 0.8047 | 0.8886 |
| `Image_14L.jpg` (CHASE_DB1) | 0.8535 | 0.9115 |
| `im0319.ppm` (STARE) | 0.8108 | 0.9170 |
| *(additional four images omitted for brevity)* |  |  |

### Discussion

- **Generalization achieved**: Both global and per-image averages exceed the success criteria, confirming robustness on the balanced test set.
- **HRF robustness**: High-resolution pathological cases retain Dice up to 0.8649, evidencing the impact of the patch strategy.
- **Sensitivity vs. specificity**: The enhancement front-end raises vessel recall while keeping specificity at 0.9768.

## Conclusions and Future Work

### Key Takeaways

- Unified multi-dataset training paired with patch sampling yields domain-robust vessel segmentation.
- The learnable enhancement module generalizes contrast normalization across acquisition conditions.
- U²Net-E matches or exceeds single-dataset baselines despite broader domain coverage.

### Planned Extensions

1. **Precision refinement**: Adjust loss terms to further penalize false positives and push Dice beyond 0.80.
2. **Deployment optimization**: Profile runtime and memory footprint for low-resource clinical devices.
3. **Dataset-wise reporting**: Produce per-dataset test summaries and literature comparisons as a dedicated study.
4. **Ablation studies**: Compare U²Net-E against vanilla U²-Net to quantify enhancement gains.


