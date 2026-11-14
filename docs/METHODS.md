# Methods (`v2`)

This document describes the backbone, the enhancement front-ends and the training recipe.
For the evaluation protocol see [PROTOCOL.md](PROTOCOL.md), for commands see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md), and for results see [RESULTS.md](RESULTS.md).

Every experiment uses the **same backbone with unchanged code**. Only three things differ between
variants: the input front-end, the image scale and the augmentation.

---

## 1. Backbone: U²-Net-lite

The lightweight configuration of U²-Net (Qin et al., 2020), 1.14–1.17 M parameters depending on the
front-end. Six encoder stages and five decoder stages, each a ReSidual U-block (RSU).

| Encoder | L | in | mid | out |   | Decoder | L | in | mid | out |
| :--- | :-: | :-: | :-: | :-: | :-: | :--- | :-: | :-: | :-: | :-: |
| En_1 | 7 | 3 | 16 | 64 | | De_1 | 7 | 128 | 16 | 64 |
| En_2 | 6 | 64 | 16 | 64 | | De_2 | 6 | 128 | 16 | 64 |
| En_3 | 5 | 64 | 16 | 64 | | De_3 | 5 | 128 | 16 | 64 |
| En_4 | 4 | 64 | 16 | 64 | | De_4 | 4 | 128 | 16 | 64 |
| En_5 | 4 (dilated) | 64 | 16 | 64 | | De_5 | 4 (dilated) | 128 | 16 | 64 |
| En_6 | 4 (dilated) | 64 | 16 | 64 | | | | | | |

Each decoder stage and the deepest encoder stage emit a one-channel side map; the six side maps are
upsampled and fused by a 1×1 convolution into a seventh. All seven are supervised during training
(deep supervision); **prediction uses the fused map (`d0`) only**.

---

## 2. Enhancement front-ends

### 2.1 `hybrid` — the original module (~37 k parameters)

For an RGB patch `x` in [0, 1], the green channel `g = x_G` passes through four steps.

**1. Per-patch min–max stretch**

```
ĝ = (g − min_P g) / (max_P g − min_P g + ε),      ε = 1e−5
```

taken over the whole 384 px patch `P`.

**2. Residual CNN.** A 3×3 convolution to 32 channels, two residual units of two 3×3 convolutions
each, and a 1×1 output convolution add a learned residual, `r = ĝ + f_θ(ĝ)`. Receptive field ≈ 11 px.

**3. Learnable contrast curve.** With a local mean `μ = ½(k₃ * r) + ½(k₇ * r)` and `d = r − μ`:

```
y = α · sign(d) · |d|^γ + μ + β
```

where `γ, α, β` are learned scalars (initialised 1, 1, 0) **shared by every image**.

**4. Output.** `y` is copied to three channels and mapped to [−1, 1] by `(y − 0.5) / 0.5`.

The min–max stretch makes the module invariant to a uniform gain on the patch, which is why it
resists darkening. It has two side effects that matter for the results:

- Each patch is stretched on its own, so the same pixel is mapped differently depending on which
  patch it falls in. This leaves **rectangular seams** in whole-image output.
- The minimum and maximum include the black border and the brightest structure, so a **bright optic
  disc or exudate sets the stretch for the whole patch**.

### 2.2 EnhancementV2 (~7.5 k parameters)

Designed to remove those causes: per-image 1st–99th percentile stretch inside the FOV; a learnable
1×1 colour mix initialised to green; Retinex illumination removal by division by a FOV-masked
Gaussian blur with learnable σ (initialised 48 px); a per-image tone curve in the style of Zero-DCE;
a zero-initialised residual refinement; and optionally a consistency loss requiring two colour,
exposure or lighting versions of a patch to enhance alike. The whole image is enhanced once at test
time, so no seams can appear.

### 2.3 EnhancementV3

Designed after the external test and changing **four things at once** relative to `hybrid`:

1. normalization statistics computed inside the true eye mask;
2. outlier-capped multi-scale Weber normalization (background estimated at 16/48/128 px; pixels
   brighter than the background soft-capped at `0.35·L₀ + 0.03` through a tanh; illumination
   re-estimated from the capped image);
3. a learnable noise floor `φ ∈ [0.02, 0.15]` in the denominator of the Weber contrast;
4. extra input channels (enhanced channel, a Weber map of red, and a learned blend with raw
   intensity). The `enh3_mono` ablation feeds the enhanced channel three times instead.

### 2.4 `hybrid_weber` — the final model

Because V3 changed four things at once, stage 6 added **one change at a time** to `hybrid`, keeping
everything else identical. Each change is a switch that is off by default, so existing checkpoints
load unchanged.

| Variant | The one change |
| :--- | :--- |
| `hybrid_masknorm` | per-patch min–max computed inside the eye only, maximum capped at the 99.5th percentile |
| `hybrid_rg` | raw green and red fed to the backbone beside the enhanced channel |
| `hybrid_floor` | a fixed floor of 0.05 added to the min–max denominator |
| `hybrid_weber` | per-patch min–max replaced by multi-scale Weber normalization |

`hybrid_weber` replaces the min–max stretch with

```
L  = Σ_s w_s · (G_σs * g) / (G_σs * 1),     σ ∈ {16, 48, 128} px,  w = softmax(ℓ)
c  = (g − L) / (L + 1e−3)
ĝ  = ½ + ½ · tanh(k · c),                   k ∈ [1, 12], initialised 4
```

`L` is a weighted sum of normalized Gaussian blurs at three scales, with learnable weights
initialised to 0.25 / 0.50 / 0.25. The Weber contrast `c` measures how much darker or brighter a
pixel is than its local background, relative to that background; a uniform gain on `g` cancels, so
the exposure invariance of the min–max stretch is kept while the dependence on patch extremes is
removed. Blurs are computed on downsampled grids for speed. Unlike EnhancementV3, this version has
**no eye mask, no outlier cap and no noise floor**, and `L` is computed over the whole patch
including the black border.

---

## 3. Variants

Names combine the front-end with two optional suffixes: `_scale` resizes every image so the eye is
960 px wide, `_strong` uses strong augmentation.

| Variant | Front-end | Scale norm. | Augmentation | Stages |
| :--- | :--- | :-: | :-: | :-: |
| `rgb` | none | no | basic | 1b–6b |
| `clahe` | CLAHE (green) | no | basic | 1b–5 |
| `hybrid` | original module | no | basic | 1–6b |
| `hybrid_strong` | original module | no | strong | 1b |
| `hybrid_scale` | original module | yes | basic | 1b |
| `rgb_scale_strong` | none | yes | strong | 1 |
| `clahe_scale_strong` | CLAHE (green) | yes | strong | 1 |
| `hybrid_scale_strong` | original module | yes | strong | 1 |
| `hybrid_scale_strong_fovnorm` | original + per-image FOV normalization | yes | strong | 1 |
| `enh2c_scale_strong` | EnhancementV2 + consistency loss | yes | strong | 1 |
| `enh3`, `enh3_mono` | EnhancementV3 | no | basic | 4–5 |
| `hybrid_masknorm`, `hybrid_rg`, `hybrid_floor` | original, one change | no | basic | 6 |
| `hybrid_weber` | original + Weber normalization | no | basic | 6, 6b |

---

## 4. Training recipe

Identical for every variant.

| Setting | Value |
| :--- | :--- |
| Loss | `0.5·BCE + 0.5·(1 − Tversky)`, α = 0.3 (false positives), β = 0.7 (false negatives), inside the FOV, applied to all seven outputs |
| Optimizer | Adam, lr 3e−4, weight decay 1e−5, 500-iteration warm-up, cosine decay |
| Length | up to 12,000 iterations, batch 8, 384×384 patches |
| Checkpoint | best validation Dice (no TTA), checked every 1,000 iterations; early stop after 6 checks without gain |
| Sampling | each dataset drawn with equal probability; patch centres inside the FOV |
| Basic augmentation | flips, 90° rotations, ±20° rotation, scale 0.8–1.25, brightness, blur, noise |
| Strong augmentation | basic plus colour gains, hue and saturation, contrast, gamma, illumination gradient, blur, resolution loss, noise, JPEG |

The Tversky weights favour sensitivity, which lowers precision for every variant equally; precision
is therefore reported beside every Dice value.
