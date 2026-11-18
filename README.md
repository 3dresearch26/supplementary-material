# PosDoF Supplementary Material

This repository contains the implementation and dataset accompanying the PosDoF submission.

## Contents

```text
supplementary/
├── code/
│   ├── PosDoF/       # PosDoF implementation
│   └── Coverage/     # PosDoF with coverage-aware insertion
├── examples/
├── scripts/
├── LICENSE
└── README.md
```

The complete image dataset is distributed through the `dataset-v1` GitHub Release as `PosDoF_Dataset.zip`.

## Dataset

The released dataset contains five real scenes and five synthetic scenes with fixed training and test splits.

```text
PosDoF_Dataset/
├── Real/
│   ├── RedRoom/{train,test}
│   ├── GreenRoom/{train,test}
│   ├── OrangeRoom/{train,test}
│   ├── PurpleRoom/{train,test}
│   └── YellowRoom/{train,test}
└── Synthetic/
    ├── Room/{train,test}
    ├── Lounge/{train,test}
    ├── Office/{train,test}
    ├── Cafe/{train,test}
    └── Gallery/{train,test}
```

### Real scenes

| Scene | Train | Test |
|---|---:|---:|
| RedRoom | 58 | 3 |
| GreenRoom | 58 | 3 |
| OrangeRoom | 58 | 3 |
| PurpleRoom | 57 | 3 |
| YellowRoom | 58 | 3 |

Each synthetic scene contains 42 training images and 6 fixed test images.

The fixed test images are excluded from MVS reconstruction, surface/depth estimation, and Gaussian training. They are reserved exclusively for evaluation.

## PosDoF implementation

SparseSurf optimises a Gaussian centre as a free three-dimensional parameter. PosDoF associates Gaussian \(i\) with an MVS surface anchor \(a_i\) and unit surface normal \(n_i\), restricting its admissible centre positions to the anchor-normal line

```text
C_i = { a_i + d n_i | d in R }.
```

The Gaussian centre remains a three-dimensional optimisable parameter. For centre gradient \(g_i\), PosDoF constructs the normal-direction projector

```text
P_i = n_i n_i^T
```

and retains the normal component of the positional gradient:

```text
g_i <- P_i g_i.
```

After the optimiser step, the centre is projected back onto its anchor-normal line:

```text
x_i <- a_i + P_i (x_i - a_i).
```

The positional constraint acts on the Gaussian centre. Scale, rotation, opacity, spherical-harmonic coefficients, rasterisation, and the SparseSurf loss terms retain the underlying SparseSurf formulation.

SparseSurf's image-space densification criterion is retained. When densification creates a child Gaussian, its positional handling remains consistent with the surface-constrained representation.

## Coverage-aware implementation

`code/Coverage/` contains the coverage-aware extension. It supplements the existing densification process by introducing surface-supported Gaussians in regions where the current representation provides insufficient coverage.

For pixel \(u\), the mean RGB reconstruction residual is

```text
E_rgb(u) = (1/3) sum_c |I_c(u) - I_hat_c(u)|.
```

Using the accumulated rendered opacity `alpha(u)`, the coverage-deficit score is

```text
S(u) = E_rgb(u) [1 - alpha(u)].
```

Candidate selection uses an adaptive interquartile-range threshold:

```text
tau = Q3 + 1.5 (Q3 - Q1).
```

Pixels with sufficiently high coverage-deficit scores and valid stereo depth are considered as candidates for additional Gaussian support. Candidate geometry is obtained from the available stereo/depth information and transformed into the scene coordinate system before insertion.

The coverage-aware extension and PosDoF address different parts of optimisation. PosDoF constrains positional optimisation of Gaussian centres using local surface geometry, while coverage-aware insertion introduces additional Gaussian support in under-represented image regions.

## Environment

Each implementation contains its environment specification and required source dependencies.

For PosDoF:

```bash
cd code/PosDoF
conda env create -f environment.yml
conda activate sparsesurf_bw
```

The coverage-aware implementation is provided under:

```bash
cd code/Coverage
```

and uses the corresponding environment and dependencies included in that directory.

The training pipeline requires a FoundationStereo checkpoint. Its location is supplied using:

```text
--foundation_stereo_ckpt
```

## Preparing the released images

The dataset release provides the original fixed training and test image splits. The training code expects a SparseSurf-prepared scene containing the camera registration and geometry required by the SparseSurf pipeline.

The released `test/` images must **not** be included in COLMAP/MVS reconstruction, surface/depth estimation, or Gaussian training. They are reserved for final evaluation.

After preparing a scene from the corresponding `train/` images, define:

```bash
SCENE=/path/to/prepared_scene
MODEL=/path/to/output_model
CKPT=/path/to/foundationstereo_checkpoint
```

`SCENE` should point to the prepared SparseSurf scene, `MODEL` specifies the output directory, and `CKPT` points to the FoundationStereo checkpoint.

## Train PosDoF

Enter the PosDoF implementation directory:

```bash
cd code/PosDoF
```

Set the number of training views for the selected scene. For example:

```bash
N=58
```

Train using the 7,000-iteration configuration used in the experiments:

```bash
python -u train.py \
    -s "$SCENE" \
    -m "$MODEL" \
    -r 1 \
    --eval \
    --n_views "$N" \
    --iterations 7000 \
    --test_iterations 7000 \
    --save_iterations 7000 \
    --total_virtual_num 240 \
    --foundation_stereo_ckpt "$CKPT"
```

Use the following training-view counts:

| Scene | `N` |
|---|---:|
| RedRoom | 58 |
| GreenRoom | 58 |
| OrangeRoom | 58 |
| PurpleRoom | 57 |
| YellowRoom | 58 |
| Room | 42 |
| Lounge | 42 |
| Office | 42 |
| Cafe | 42 |
| Gallery | 42 |

The final Gaussian representation is saved under the model directory, including:

```text
point_cloud/iteration_7000/point_cloud.ply
```

## Render

Render the trained representation using:

```bash
python -u render.py \
    -s "$SCENE" \
    -m "$MODEL" \
    -r 1 \
    --eval \
    --n_views "$N" \
    --iteration 7000 \
    --render_depth
```

## Evaluate image quality

After rendering, run:

```bash
python metrics.py -m "$MODEL"
```

The evaluation pipeline reports PSNR, SSIM, LPIPS, and AVGE.

## Train the coverage-aware variant

The coverage-aware variant uses the same prepared scene, fixed camera split, training schedule, and evaluation protocol.

Enter the Coverage implementation directory:

```bash
cd code/Coverage
```

Set `SCENE`, `MODEL`, `CKPT`, and `N` as above, then train:

```bash
python -u train.py \
    -s "$SCENE" \
    -m "$MODEL" \
    -r 1 \
    --eval \
    --n_views "$N" \
    --iterations 7000 \
    --test_iterations 7000 \
    --save_iterations 7000 \
    --total_virtual_num 240 \
    --foundation_stereo_ckpt "$CKPT"
```

Render the resulting model using:

```bash
python -u render.py \
    -s "$SCENE" \
    -m "$MODEL" \
    -r 1 \
    --eval \
    --n_views "$N" \
    --iteration 7000 \
    --render_depth
```

Then evaluate:

```bash
python metrics.py -m "$MODEL"
```

## Sparse-view RedRoom experiments

The RedRoom sparse-view experiments use the following numbers of training views:

```text
12, 24, 36, 58
```

The same three fixed RedRoom test images are retained for every training density. Set the required training-view count using `--n_views`.

The held-out test images remain excluded from MVS reconstruction and training for every density.

## Experimental protocol

SparseSurf, PosDoF, and the coverage-aware variant are compared using the same camera splits, training schedule, rendering configuration, and evaluation pipeline. Models are trained for 7,000 iterations.

No post-training pruning or compression is applied when reporting Gaussian counts. PosDoF does not introduce a Gaussian-count loss, an additional pruning rule, or a fixed population target.

## DTU evaluation

The DTU experiments use the 7,000-iteration PosDoF pipeline. The prepared DTU configuration contains three reconstruction views per scan and does not contain an independent held-out novel-view test set.

Accordingly, DTU image-quality measurements correspond to reconstruction/input-view quality and should not be interpreted as held-out novel-view results.

## Dataset download

The complete dataset is available from the `dataset-v1` release:

https://github.com/3dresearch26/supplementary-material/releases/tag/dataset-v1

Download `PosDoF_Dataset.zip` from the release assets.

## Anonymous review

This repository is provided as anonymous supplementary material for double-blind peer review.
