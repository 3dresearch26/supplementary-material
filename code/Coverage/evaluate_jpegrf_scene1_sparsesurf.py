from pathlib import Path

import csv
import math

import lpips
import numpy as np
import torch

from PIL import Image
from pytorch_msssim import ssim


ROOT = Path(
    "outputs/jpegrf_scene1_24views_sparsesurf_7000/test/ours_7000"
)

GT_DIR = ROOT / "gt"
RENDER_DIR = ROOT / "renders"


VIEWS = [
    "frame_000021.png",
    "frame_000138.png",
    "frame_000217.png",
]


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Device:", DEVICE)


# Use LPIPS-VGG consistently with the other 3DGS-style
# benchmark evaluations.
lpips_model = (
    lpips.LPIPS(net="vgg")
    .to(DEVICE)
)

lpips_model.eval()


def load_image(path):

    image = (
        Image.open(path)
        .convert("RGB")
    )

    array = (
        np.asarray(image)
        .astype(np.float32)
        / 255.0
    )

    tensor = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    return tensor.to(DEVICE)


def calc_psnr(pred, gt):

    mse = torch.mean(
        (pred - gt) ** 2
    ).item()

    if mse == 0:
        return float("inf")

    return (
        -10.0
        * math.log10(mse)
    )


rows = []


print()
print("=" * 70)
print(
    "SparseSurf — JPEG-RF Scene 1 "
    "— 24 views — iteration 7000"
)
print("=" * 70)


for name in VIEWS:

    gt_path = GT_DIR / name
    render_path = RENDER_DIR / name

    if not gt_path.exists():
        raise FileNotFoundError(
            gt_path
        )

    if not render_path.exists():
        raise FileNotFoundError(
            render_path
        )

    gt = load_image(gt_path)
    pred = load_image(render_path)

    if gt.shape != pred.shape:
        raise RuntimeError(
            f"Shape mismatch for {name}: "
            f"GT={tuple(gt.shape)}, "
            f"Render={tuple(pred.shape)}"
        )

    with torch.no_grad():

        psnr_value = calc_psnr(
            pred,
            gt
        )

        ssim_value = ssim(
            pred,
            gt,
            data_range=1.0,
            size_average=True
        ).item()

        # LPIPS expects [-1, 1].
        lpips_value = lpips_model(
            pred * 2.0 - 1.0,
            gt * 2.0 - 1.0
        ).item()


    row = {
        "view": name,
        "PSNR": psnr_value,
        "SSIM": ssim_value,
        "LPIPS_VGG": lpips_value,
    }

    rows.append(row)


    print(
        f"{name:22s} "
        f"PSNR={psnr_value:8.4f}  "
        f"SSIM={ssim_value:.6f}  "
        f"LPIPS-VGG={lpips_value:.6f}"
    )


mean_psnr = np.mean(
    [r["PSNR"] for r in rows]
)

mean_ssim = np.mean(
    [r["SSIM"] for r in rows]
)

mean_lpips = np.mean(
    [r["LPIPS_VGG"] for r in rows]
)


print()
print("-" * 70)

print(
    f"{'MEAN':22s} "
    f"PSNR={mean_psnr:8.4f}  "
    f"SSIM={mean_ssim:.6f}  "
    f"LPIPS-VGG={mean_lpips:.6f}"
)

print("-" * 70)


csv_path = (
    ROOT /
    "jpegrf_scene1_sparsesurf_metrics_7000.csv"
)


with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "view",
            "PSNR",
            "SSIM",
            "LPIPS_VGG",
        ],
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(row)

    writer.writerow(
        {
            "view": "MEAN",
            "PSNR": mean_psnr,
            "SSIM": mean_ssim,
            "LPIPS_VGG": mean_lpips,
        }
    )


print()
print("Saved:")
print(csv_path)
