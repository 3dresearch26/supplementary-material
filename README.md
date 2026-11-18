# PosDoF Supplementary Material

This repository contains the implementation and dataset accompanying the anonymous PosDoF submission.

## Contents

```text
supplementary/
├── code/
│   ├── PosDoF/       # main PosDoF implementation
│   └── Coverage/     # PosDoF + coverage-aware insertion
├── examples/
├── scripts/
├── LICENSE
└── README.md
```

The complete image dataset is distributed through the `dataset-v1` GitHub Release as `PosDoF_Dataset.zip`.

## Dataset

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

Real-scene splits:

| Scene | Train | Test |
|---|---:|---:|
| RedRoom | 58 | 3 |
| GreenRoom | 58 | 3 |
| OrangeRoom | 58 | 3 |
| PurpleRoom | 57 | 3 |
| YellowRoom | 58 | 3 |

Each synthetic scene contains 42 training images and 6 fixed test images.

The fixed test images are excluded from MVS reconstruction, surface/depth estimation and Gaussian training. They are used only for evaluation.

## PosDoF implementation

SparseSurf optimises a Gaussian centre as a free three-dimensional parameter. PosDoF associates Gaussian `i` with an MVS surface anchor `a_i` and unit surface normal `n_i`, restricting the admissible centre positions to

```text
C_i = { a_i + d n_i | d in R }.
```

The centre remains a 3D optimisable parameter. For centre gradient `g_i`, PosDoF uses the normal-direction projector

```text
P_i = n_i n_i^T
```

and retains

```text
g_i <- P_i g_i.
```

After the optimiser step, the centre is projected back to its anchor-normal line:

```text
x_i <- a_i + P_i (x_i - a_i).
```

Only Gaussian position is constrained. Scale, rotation, opacity, spherical-harmonic coefficients, rasterisation and the SparseSurf loss terms are unchanged.

SparseSurf's original image-space densification criterion is retained. When cloning or splitting creates a child Gaussian, its birth position becomes its new anchor and it inherits the parent surface normal.

## Coverage-aware implementation

`code/Coverage/` contains the optional coverage-aware extension. It supplements the existing densification process by adding surface-supported Gaussians where the current representation has insufficient support.

For pixel `u`, the mean RGB reconstruction residual is

```text
E_rgb(u) = (1/3) sum_c |I_c(u) - I_hat_c(u)|.
```

With accumulated rendered opacity `alpha(u)`, the coverage-deficit score is

```text
S(u) = E_rgb(u) [1 - alpha(u)].
```

Candidate pixels are selected with an adaptive IQR threshold:

```text
tau = Q3 + 1.5 (Q3 - Q1).
```

Pixels satisfying `S(u) > tau` and having valid stereo depth can generate new Gaussians. Their positions are obtained by back-projecting the stereo depth. The reconstructed point becomes the new Gaussian anchor, after which the Gaussian follows the same PosDoF positional constraint.

The coverage extension is separate from the main method: PosDoF changes how existing Gaussian centres move, while coverage-aware insertion changes where additional Gaussian support is introduced.

## Environment

Each implementation contains its environment specification and required source dependencies.

For PosDoF:

```bash
cd code/PosDoF
conda env create -f environment.yml
conda activate sparsesurf_bw
```

For the coverage variant, use the corresponding `code/Coverage` directory with the same software environment.

A FoundationStereo checkpoint is required by the training pipeline and is passed through `--foundation_stereo_ckpt`.

## Preparing the released images

The release intentionally exposes the original train/test image split. The training code expects a SparseSurf-prepared scene rather than training directly from the two image folders.

Before training, prepare the scene with the SparseSurf/COLMAP/MVS pipeline so that camera registration and the required stereo/MVS geometry are available.

**Do not include the released `test/` images in MVS or training.**

After preparation, define:

```bash
SCENE=/path/to/prepared_scene
MODEL=/path/to/output_model
CKPT=/path/to/foundationstereo_checkpoint
```

## Train PosDoF

```bash
cd code/PosDoF
```

Set the number of training views. For example, RedRoom uses:

```bash
N=58
```

Train for the 7,000-iteration schedule used in the experiments:

```bash
python -u train.py     -s "$SCENE"     -m "$MODEL"     -r 1     --eval     --n_views "$N"     --iterations 7000     --test_iterations 7000     --save_iterations 7000     --total_virtual_num 240     --foundation_stereo_ckpt "$CKPT"
```

Use:

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

The final representation is saved under the model directory, including:

```text
point_cloud/iteration_7000/point_cloud.ply
```

## Render

```bash
python -u render.py     -s "$SCENE"     -m "$MODEL"     -r 1     --eval     --n_views "$N"     --iteration 7000     --render_depth
```

## Evaluate image quality

After rendering:

```bash
python metrics.py -m "$MODEL"
```

The evaluation pipeline reports PSNR, SSIM and LPIPS and also computes AVGE.

## Train the coverage-aware variant

Use the same prepared scene, camera split, training schedule and evaluation pipeline:

```bash
cd code/Coverage
```

Then:

```bash
python -u train.py     -s "$SCENE"     -m "$MODEL"     -r 1     --eval     --n_views "$N"     --iterations 7000     --test_iterations 7000     --save_iterations 7000     --total_virtual_num 240     --foundation_stereo_ckpt "$CKPT"
```

Render and evaluate using the same commands:

```bash
python -u render.py     -s "$SCENE"     -m "$MODEL"     -r 1     --eval     --n_views "$N"     --iteration 7000     --render_depth

python metrics.py -m "$MODEL"
```

## Sparse-view RedRoom experiments

The fixed three RedRoom test images remain unchanged while the training-view count is varied:

```text
12, 24, 36, 58
```

Set the corresponding value through `--n_views`. The held-out images must remain excluded from MVS and training at every density.

## Experimental protocol

For SparseSurf, PosDoF and PosDoF with coverage, comparisons use the same camera splits, training schedule, rendering configuration and evaluation pipeline. Models are trained for 7,000 iterations.

No post-training pruning or compression is applied when reporting Gaussian counts. PosDoF does not use a Gaussian-count loss, an additional pruning rule or a fixed population target.

## DTU note

The public DTU evaluation uses the same 7,000-iteration PosDoF pipeline. The prepared DTU configuration contains three reconstruction views per scan and no independent held-out novel-view test images. DTU image metrics therefore represent reconstruction/input-view quality, not held-out novel-view quality.

## Dataset download

Dataset release:

https://github.com/3dresearch26/supplementary/releases/tag/dataset-v1

Download `PosDoF_Dataset.zip` from the release assets.

## Anonymous review

This repository is provided as anonymous supplementary material for double-blind peer review. Author and institutional identifying information is intentionally omitted.
